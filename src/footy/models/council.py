"""
Expert Council Model — Multi-expert ensemble for football match prediction.

Architecture:
    Layer 1 — 34 specialist experts (see footy.models.experts package)
    Layer 2 — Conflict, consensus, KL divergence & interaction signals across experts
    Layer 3 — Multi-model stack (HistGBM + RF + LR) with LEARNED weights + isotonic cal
    Layer 4 — Walk-forward cross-validation for honest performance estimation
    Layer 5 — Ollama interpreter (optional) for narrative match analysis

Expert modules live in ``footy.models.experts``.  This module contains the
meta-learner, SQL queries, training pipeline, and prediction logic.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from footy.models.advanced_math import (
    kl_divergence, log_transform, tanh_transform,
    jensen_shannon_divergence, multi_expert_jsd,
)
from footy.config import settings
from footy.normalize import canonical_team_name
from footy.models.dixon_coles import DCModel, fit_dc, predict_1x2

# Re-export all expert symbols for backward compatibility
from footy.models.experts import (  # noqa: F401
    ALL_EXPERTS,
    Expert,
    ExpertResult,
    BayesianRateExpert,
    ContextExpert,
    EloExpert,
    FormExpert,
    GoalPatternExpert,
    H2HExpert,
    InjuryAvailabilityExpert,
    LeagueTableExpert,
    MarketExpert,
    MomentumExpert,
    PoissonExpert,
    _entropy3,
    _f,
    _implied,
    _label,
    _norm3,
    _pts,
    _raw,
)


# ---------------------------------------------------------------------------
# Expert lookup helper — Replace hardcoded indices with name-based lookup
# ---------------------------------------------------------------------------

def _expert_by_name(experts, name):
    """Get expert result by name from list of (expert, result) pairs or just results.

    Args:
        experts: List of (Expert, ExpertResult) tuples OR list of ExpertResult
        name: Name of expert to find (e.g., "EloExpert", "PoissonExpert")

    Returns:
        ExpertResult or None if not found
    """
    if not experts:
        return None
    # Handle case where we have (expert, result) tuples
    if isinstance(experts[0], tuple) and len(experts[0]) == 2:
        for exp, res in experts:
            if exp.__class__.__name__ == name or exp.name == name:
                return res
    return None


# ---------------------------------------------------------------------------
# Expert index-based access (maintained for backward compatibility)
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

MODEL_VERSION = "v12_analyst"
MODEL_PATH = Path("data/models") / f"{MODEL_VERSION}.joblib"

# ---------------------------------------------------------------------------
# Shared SQL column / join fragments — single source of truth
# ---------------------------------------------------------------------------
_MATCH_COLS = """\
m.utc_date, m.competition, m.home_team, m.away_team,
               m.home_goals, m.away_goals,
               e.b365h, e.b365d, e.b365a, e.raw_json,
               e.b365ch, e.b365cd, e.b365ca,
               e.psh, e.psd, e.psa,
               e.avgh, e.avgd, e.avga,
               e.maxh, e.maxd, e.maxa,
               e.b365_o25, e.b365_u25,
               e.avg_o25, e.avg_u25,
               e.max_o25, e.max_u25,
               e.hs, e.hst, e.hc, e.hy, e.hr,
               e.as_ AS as_, e.ast, e.ac, e.ay, e.ar,
               e.hthg, e.htag,
               e.odds_ah_line, e.odds_ah_home, e.odds_ah_away,
               e.odds_btts_yes, e.odds_btts_no,
               e.formation_home, e.formation_away,
               e.af_xg_home, e.af_xg_away,
               e.af_possession_home, e.af_possession_away,
               fpl_h.injury_score AS fpl_inj_score_h,
               fpl_h.squad_strength AS fpl_squad_str_h,
               fpl_h.available AS fpl_available_h,
               fpl_h.doubtful AS fpl_doubtful_h,
               fpl_h.injured AS fpl_injured_h,
               fpl_h.suspended AS fpl_suspended_h,
               fpl_a.injury_score AS fpl_inj_score_a,
               fpl_a.squad_strength AS fpl_squad_str_a,
               fpl_a.available AS fpl_available_a,
               fpl_a.doubtful AS fpl_doubtful_a,
               fpl_a.injured AS fpl_injured_a,
               fpl_a.suspended AS fpl_suspended_a,
               fdr_h.fdr_next_3 AS fpl_fdr_h,
               fdr_a.fdr_next_3 AS fpl_fdr_a,
               fdr_h.fdr_next_6 AS fpl_fdr6_h,
               fdr_a.fdr_next_6 AS fpl_fdr6_a,
               af.home_injuries AS af_inj_h,
               af.away_injuries AS af_inj_a,
               wd.kickoff_temperature_c AS wx_temp_c,
               wd.kickoff_precipitation_mm AS wx_precip_mm,
               wd.kickoff_wind_speed_kmh AS wx_wind_kmh,
               wd.kickoff_wind_gusts_kmh AS wx_gusts_kmh,
               wd.kickoff_humidity_pct AS wx_humidity_pct,
               wd.kickoff_cloud_cover_pct AS wx_cloud_cover_pct,
               wd.rainfall_prev_24h_mm AS wx_rain_24h_mm,
               wd.rainfall_prev_48h_mm AS wx_rain_48h_mm,
               e.understat_xg_home, e.understat_xg_away,
               e.referee_name,
               mv_h.squad_market_value_eur AS mv_home_squad_value,
               mv_h.average_player_value_eur AS mv_home_avg_value,
               mv_h.squad_size AS mv_home_depth,
               mv_h.average_age AS mv_home_avg_age,
               mv_a.squad_market_value_eur AS mv_away_squad_value,
               mv_a.average_player_value_eur AS mv_away_avg_value,
               mv_a.squad_size AS mv_away_depth,
               mv_a.average_age AS mv_away_avg_age,
               vs.home_advantage_strength AS venue_ha_strength,
               vs.avg_home_scored AS venue_avg_home_scored,
               vs.avg_home_conceded AS venue_avg_home_conceded,
               vs.home_clean_sheet_pct AS venue_cs_pct,
               stm.capacity AS venue_capacity,
               stm.altitude_m AS venue_altitude_m,
               stm.latitude AS venue_lat,
               stm.longitude AS venue_lon,
               stm.surface AS venue_surface,
               ra.referee_name AS ref_assigned_name,
               ra.yellow_cards_per_match AS ref_yellow_per_match,
               ra.red_cards_per_match AS ref_red_per_match,
               ra.penalties_per_match AS ref_penalty_per_match,
               ra.home_bias_ratio AS ref_home_bias_ratio,
               ra.historical_matches AS ref_historical_matches"""

_MATCH_JOINS = """\
FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        LEFT JOIN fpl_availability fpl_h ON fpl_h.team = m.home_team
        LEFT JOIN fpl_availability fpl_a ON fpl_a.team = m.away_team
        LEFT JOIN fpl_fixture_difficulty fdr_h ON fdr_h.team = m.home_team
        LEFT JOIN fpl_fixture_difficulty fdr_a ON fdr_a.team = m.away_team
        LEFT JOIN af_context af ON af.match_id = m.match_id
        LEFT JOIN weather_data wd ON wd.match_id = m.match_id
        LEFT JOIN market_values mv_h ON mv_h.team = m.home_team
        LEFT JOIN market_values mv_a ON mv_a.team = m.away_team
        LEFT JOIN venue_stats vs ON vs.team = m.home_team
        LEFT JOIN stadiums stm ON stm.team = m.home_team
        LEFT JOIN referee_assignments ra ON ra.match_id = m.match_id"""


def _match_query(where: str, *, include_match_id: bool = False) -> str:
    """Build a match query with standard columns, joins, and given WHERE clause."""
    id_col = "m.match_id, " if include_match_id else ""
    return f"SELECT {id_col}{_MATCH_COLS}\n        {_MATCH_JOINS}\n        WHERE {where}\n        ORDER BY m.utc_date ASC"


def _run_experts(df: pd.DataFrame, experts: list[Expert] | None = None) -> list[ExpertResult]:
    """Run all experts on a DataFrame of matches."""
    if experts is None:
        experts = ALL_EXPERTS
    return [expert.compute(df) for expert in experts]


def _build_meta_X(results: list[ExpertResult], experts: list[Expert] | None = None,
                  competitions: np.ndarray | None = None) -> np.ndarray:
    """Construct the meta-feature matrix from expert results.

    Layout:
        - Expert probs:          N × 3  (N = current council expert count)
        - Expert confidence:     N
        - Expert domain features: ~200+
        - Conflict signals:      ~15
        - Per-expert entropy:    N
        - Per-expert KL div:     N
        - Feature interactions:  ~20
        - Upset-detection:       ~30
        - Competition encoding:  5 (one-hot)
        ≈ 400+ total columns
    """
    if experts is None:
        experts = ALL_EXPERTS
    n = results[0].probs.shape[0]
    blocks: list[np.ndarray] = []

    # 1. expert probabilities + confidence
    for res in results:
        blocks.append(res.probs)               # (n, 3)
        blocks.append(res.confidence[:, None])  # (n, 1)

    # 2. domain-specific features
    for res in results:
        for arr in res.features.values():
            blocks.append(arr[:, None] if arr.ndim == 1 else arr)

    # 3. conflict / consensus signals
    all_probs = np.stack([r.probs for r in results], axis=0)  # (N_experts, n, 3)
    # variance across experts per outcome
    var_h = np.var(all_probs[:, :, 0], axis=0)
    var_d = np.var(all_probs[:, :, 1], axis=0)
    var_a = np.var(all_probs[:, :, 2], axis=0)
    # max - min spread
    spread_h = np.max(all_probs[:, :, 0], axis=0) - np.min(all_probs[:, :, 0], axis=0)
    spread_a = np.max(all_probs[:, :, 2], axis=0) - np.min(all_probs[:, :, 2], axis=0)
    # average confidence
    avg_conf = np.mean([r.confidence for r in results], axis=0)
    # entropy of the ensemble mean
    mean_probs = np.nan_to_num(np.mean(all_probs, axis=0), nan=1.0/3)
    ens_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)
    # weighted consensus (confidence-weighted mean probability)
    weights = np.stack([r.confidence for r in results], axis=0)  # (N_experts, n)
    weights_sum = weights.sum(axis=0, keepdims=True) + 1e-12
    w_normed = weights / weights_sum
    weighted_probs = np.sum(
        all_probs * w_normed[:, :, None], axis=0
    )  # (n, 3)

    blocks.extend([
        var_h[:, None], var_d[:, None], var_a[:, None],
        spread_h[:, None], spread_a[:, None],
        avg_conf[:, None], ens_entropy[:, None],
        weighted_probs,  # (n, 3) — confidence-weighted ensemble
    ])

    # 4. cross-expert interaction features (pairwise)
    # v12: Name-based expert lookup instead of hardcoded indices
    n_exp = len(results)
    _expert_map: dict[str, int] = {}
    if experts is not None:
        for idx, exp in enumerate(experts):
            _expert_map[exp.name] = idx
    else:
        for idx, exp in enumerate(ALL_EXPERTS):
            _expert_map[exp.name] = idx

    def _r(name: str) -> ExpertResult:
        """Get expert result by name, fallback to uniform."""
        idx = _expert_map.get(name)
        if idx is not None and idx < len(results):
            return results[idx]
        # Fallback: return uniform probs with zero features
        return ExpertResult(
            probs=np.full((n, 3), 1 / 3),
            confidence=np.zeros(n),
            features={},
        )

    # Pairwise agreements using name-based lookup
    elo_r = _r("elo"); mkt_r = _r("market"); form_r = _r("form")
    pois_r = _r("poisson"); h2h_r = _r("h2h"); ctx_r = _r("context")
    gp_r = _r("goal_pattern"); lt_r = _r("league_table"); mom_r = _r("momentum")
    bayes_r = _r("bayesian_rate"); kalman_r = _r("kalman_elo"); negbin_r = _r("negbin")

    elo_mkt_agree = 1.0 - np.abs(elo_r.probs[:, 0] - mkt_r.probs[:, 0])
    form_h2h_agree = 1.0 - np.abs(form_r.probs[:, 0] - h2h_r.probs[:, 0])
    pois_mkt_agree = 1.0 - np.abs(pois_r.probs[:, 0] - mkt_r.probs[:, 0])
    elo_form_agree = 1.0 - np.abs(elo_r.probs[:, 0] - form_r.probs[:, 0])
    pois_elo_agree = 1.0 - np.abs(pois_r.probs[:, 0] - elo_r.probs[:, 0])
    # v12: All agreements use name-based lookup — no index dependency
    gp_form_agree = 1.0 - np.abs(gp_r.probs[:, 0] - form_r.probs[:, 0])
    lt_elo_agree = 1.0 - np.abs(lt_r.probs[:, 0] - elo_r.probs[:, 0])
    mom_form_agree = 1.0 - np.abs(mom_r.probs[:, 0] - form_r.probs[:, 0])
    mom_elo_agree = 1.0 - np.abs(mom_r.probs[:, 0] - elo_r.probs[:, 0])
    bayes_pois_agree = 1.0 - np.abs(bayes_r.probs[:, 0] - pois_r.probs[:, 0])
    bayes_mkt_agree = 1.0 - np.abs(bayes_r.probs[:, 0] - mkt_r.probs[:, 0])
    # v12: New expert agreements
    kalman_elo_agree = 1.0 - np.abs(kalman_r.probs[:, 0] - elo_r.probs[:, 0])
    negbin_pois_agree = 1.0 - np.abs(negbin_r.probs[:, 0] - pois_r.probs[:, 0])
    # max disagreement across any pair of experts (for home win)
    home_stack = np.stack([r.probs[:, 0] for r in results])  # (n_exp, n)
    max_disagree = home_stack.max(axis=0) - home_stack.min(axis=0)
    # number of experts that agree on same winner
    all_winners = np.argmax(np.stack([r.probs for r in results]), axis=2)  # (n_exp, n)
    winner_votes = np.zeros(n)
    for ii in range(n):
        counts = np.bincount(all_winners[:, ii], minlength=3)
        winner_votes[ii] = counts.max() / n_exp
    blocks.extend([
        elo_mkt_agree[:, None], form_h2h_agree[:, None],
        pois_mkt_agree[:, None], elo_form_agree[:, None],
        pois_elo_agree[:, None], gp_form_agree[:, None],
        lt_elo_agree[:, None], mom_form_agree[:, None],
        mom_elo_agree[:, None], max_disagree[:, None],
        winner_votes[:, None],
        bayes_pois_agree[:, None], bayes_mkt_agree[:, None],
        kalman_elo_agree[:, None], negbin_pois_agree[:, None],
    ])

    # 5. Per-expert entropy (how uncertain each expert is) — vectorized
    for res in results:
        safe_probs = np.nan_to_num(res.probs, nan=1.0/3)
        expert_ent = -np.sum(safe_probs * np.log(safe_probs + 1e-12), axis=1)
        blocks.append(expert_ent[:, None])

    # 5b. v10: Per-expert KL divergence from ensemble mean — vectorized
    for res in results:
        safe_probs = np.nan_to_num(res.probs, nan=1.0/3)
        kl_arr = np.sum(safe_probs * np.log((safe_probs + 1e-12) / (mean_probs + 1e-12)), axis=1)
        kl_arr = np.clip(kl_arr, 0.0, None)  # KL >= 0
        blocks.append(kl_arr[:, None])

    # 6. Feature interactions (cross-domain signal combinations)
    # v12: Name-based feature access instead of hardcoded results[N]
    elo_diff = elo_r.features.get("elo_diff", np.zeros(n))
    form_pts_h = form_r.features.get("form_pts_h", np.zeros(n))
    form_pts_a = form_r.features.get("form_pts_a", np.zeros(n))
    form_diff = form_pts_h - form_pts_a
    blocks.append((elo_diff * form_diff)[:, None])  # Elo × Form interaction
    # Market probability × Poisson probability (product of independent estimates)
    mkt_ph = mkt_r.probs[:, 0]
    pois_ph = pois_r.probs[:, 0]
    blocks.append((mkt_ph * pois_ph)[:, None])  # Market × Poisson interaction
    # Momentum × recent form (amplifier when both agree)
    if "momentum" in _expert_map:
        mom_cross = mom_r.features.get("mom_ema_cross_diff", np.zeros(n))
        blocks.append((mom_cross * form_diff)[:, None])  # Momentum × Form
        # Goal scoring burst × opponent defensive weakness
        burst_h = mom_r.features.get("mom_burst_h", np.zeros(n))
        ga_slope_a = mom_r.features.get("mom_ga_slope_a", np.zeros(n))
        blocks.append((burst_h * ga_slope_a)[:, None])  # Attack surge × defensive leak

    # 6b. v10: Additional cross-domain interactions
    # Elo prominence features — raw Elo signals for direct meta-learner access
    elo_home_raw = elo_r.features.get("elo_home", np.zeros(n))
    elo_away_raw = elo_r.features.get("elo_away", np.zeros(n))
    blocks.append(elo_home_raw[:, None])   # raw home Elo rating
    blocks.append(elo_away_raw[:, None])   # raw away Elo rating
    blocks.append(elo_diff[:, None])       # raw Elo diff (home - away)
    # tanh(elo_diff) — nonlinear bounded elo signal
    elo_tanh = np.tanh(elo_diff.astype(float) / 400.0)
    blocks.append(elo_tanh[:, None])
    # log(form_diff) — signed log of form difference
    fd = form_diff.astype(float)
    form_log = np.sign(fd) * np.log1p(np.abs(fd))
    blocks.append(form_log[:, None])
    # Bayesian × Poisson agreement interaction
    if "bayesian_rate" in _expert_map:
        bayes_wr_diff = bayes_r.features.get("bayes_wr_diff", np.zeros(n))
        pois_lam_diff = pois_r.features.get("pois_lambda_diff", np.zeros(n))
        blocks.append((bayes_wr_diff * pois_lam_diff)[:, None])
    # DC-adjusted Poisson × Market (sharper probability estimates)
    dc_ph = pois_r.features.get("pois_dc_ph", np.zeros(n))
    blocks.append((dc_ph * mkt_ph)[:, None])  # DC-Poisson × Market
    # Monte Carlo P(home) × Market P(home)
    mc_ph_arr = pois_r.features.get("pois_mc_ph", np.zeros(n))
    blocks.append((mc_ph_arr * mkt_ph)[:, None])  # MC × Market

    # 6c. v11: New academic model features
    # Bivariate Poisson P(home) — Karlis & Ntzoufras 2003
    bp_ph = pois_r.features.get("pois_bp_ph", np.zeros(n))
    blocks.append(bp_ph[:, None])
    # Bivariate Poisson × Market (independent vs correlated model agreement)
    blocks.append((bp_ph * mkt_ph)[:, None])
    # Frank Copula P(home) — full-scoreline dependency
    cop_ph = pois_r.features.get("pois_cop_ph", np.zeros(n))
    blocks.append(cop_ph[:, None])
    # Copula × DC agreement (two dependency models agreeing)
    blocks.append((cop_ph * dc_ph)[:, None])
    # COM-Poisson P(home) — dispersion-aware
    cmp_ph = pois_r.features.get("pois_cmp_ph", np.zeros(n))
    blocks.append(cmp_ph[:, None])
    # COM-Poisson dispersion index (over-dispersed = unpredictable)
    cmp_disp_h = pois_r.features.get("pois_cmp_disp_h", np.zeros(n))
    cmp_disp_a = pois_r.features.get("pois_cmp_disp_a", np.zeros(n))
    blocks.append(cmp_disp_h[:, None])
    blocks.append(cmp_disp_a[:, None])
    # Bradley-Terry P(home) — pairwise comparison model
    bt_ph = pois_r.features.get("pois_bt_ph", np.zeros(n))
    bt_pd = pois_r.features.get("pois_bt_pd", np.zeros(n))
    blocks.append(bt_ph[:, None])
    blocks.append(bt_pd[:, None])
    # Bradley-Terry × Elo agreement (two strength-based models)
    blocks.append((bt_ph * elo_r.probs[:, 0])[:, None])
    # Model disagreement: spread across DC, Copula, BiPois, CMP models
    model_stack = np.column_stack([
        dc_ph, cop_ph, bp_ph, cmp_ph, mc_ph_arr
    ])
    model_spread = np.std(model_stack, axis=1)
    model_mean = np.mean(model_stack, axis=1)
    blocks.append(model_spread[:, None])  # model disagreement
    blocks.append(model_mean[:, None])    # model consensus

    # 6d. v11: Jensen-Shannon Divergence (symmetric, bounded)
    # Vectorized pairwise JSD: Elo-Market
    p_elo = np.nan_to_num(elo_r.probs, nan=1.0/3)  # (n, 3)
    p_mkt = np.nan_to_num(mkt_r.probs, nan=1.0/3)  # (n, 3)
    m_em = 0.5 * (p_elo + p_mkt) + 1e-12
    kl1 = np.sum(p_elo * np.log((p_elo + 1e-12) / m_em), axis=1)
    kl2 = np.sum(p_mkt * np.log((p_mkt + 1e-12) / m_em), axis=1)
    elo_mkt_jsd = np.clip(0.5 * (kl1 + kl2), 0.0, None)
    blocks.append(elo_mkt_jsd[:, None])
    # Multi-expert generalised JSD (total disagreement across ALL experts)
    all_expert_jsd = np.zeros(n)
    all_p = np.nan_to_num(np.stack([r.probs for r in results[:n_exp]]), nan=1.0/3)  # (n_exp, n, 3)
    m_all = all_p.mean(axis=0) + 1e-12  # (n, 3)
    for r in results[:n_exp]:
        safe_r = np.nan_to_num(r.probs, nan=1.0/3)
        all_expert_jsd += np.sum(safe_r * np.log((safe_r + 1e-12) / m_all), axis=1)
    all_expert_jsd = np.clip(all_expert_jsd / n_exp, 0.0, None)
    blocks.append(all_expert_jsd[:, None])

    # 6e. Upset-detection cross-expert interaction features
    # These capture factor combinations that systematically predict upsets

    # H2H bogey × market favourite — historic bogey ground + market overconfidence
    h2h_bogey = h2h_r.features.get("h2h_bogey", np.zeros(n))
    upset_bogey_mkt = h2h_bogey * mkt_ph
    blocks.append(upset_bogey_mkt[:, None])

    # H2H surprise rate × context fatigue — historic upsets + current tiredness
    h2h_surprise = h2h_r.features.get("h2h_surprise_rate", np.zeros(n))
    fatigue_diff = ctx_r.features.get("ctx_fatigue_diff", np.zeros(n))
    blocks.append((h2h_surprise * np.abs(fatigue_diff))[:, None])

    # Motivation difference × market strength — motivated underdog vs complacent favourite
    motiv_diff = ctx_r.features.get("ctx_motivation_diff", np.zeros(n))
    blocks.append((motiv_diff * mkt_ph)[:, None])

    # Home fatigue × away freshness — tired home team vs rested visitor
    fatigue_h = ctx_r.features.get("ctx_fatigue_h", np.zeros(n))
    fatigue_a = ctx_r.features.get("ctx_fatigue_a", np.zeros(n))
    blocks.append((fatigue_h * (1.0 - fatigue_a))[:, None])

    # Relegation battle flag × underdogness — desperate underdog
    is_rel_a = ctx_r.features.get("ctx_is_relegation_a", np.zeros(n))
    blocks.append((is_rel_a * (1.0 - mkt_ph))[:, None])  # away-underdog in relegation fight

    # Derby/rivalry × market certainty — derbies erode favourite advantage
    is_derby = ctx_r.features.get("ctx_is_derby", np.zeros(n))
    blocks.append((is_derby * mkt_ph)[:, None])  # higher market certainty in derby = more vulnerable
    # High-stakes × team quality gap — high-stakes matches tighten outcomes
    high_stakes = ctx_r.features.get("ctx_high_stakes", np.zeros(n))
    blocks.append((high_stakes * np.abs(elo_diff.astype(float) / 400.0))[:, None])

    # Venue form gap — home-specific form vs away-specific form
    hf_ppg_h = form_r.features.get("form_home_ppg_h", np.zeros(n))
    af_ppg_a = form_r.features.get("form_away_ppg_a", np.zeros(n))
    venue_form_gap = hf_ppg_h - af_ppg_a
    blocks.append(venue_form_gap[:, None])

    # xG overperformance — regression signal: scoring above xG = unsustainable
    xg_diff_h = form_r.features.get("form_xg_diff_h", np.zeros(n))
    xg_diff_a = form_r.features.get("form_xg_diff_a", np.zeros(n))
    blocks.append(xg_diff_h[:, None])  # positive = overperforming
    blocks.append(xg_diff_a[:, None])
    blocks.append((xg_diff_h * mkt_ph)[:, None])  # overperforming favourite

    # Form vs top opponents — quality-adjusted form signals
    form_vs_top_h = form_r.features.get("form_vs_top_h", np.zeros(n))
    form_vs_top_a = form_r.features.get("form_vs_top_a", np.zeros(n))
    blocks.append((form_vs_top_h - form_vs_top_a)[:, None])

    # Position-form gap × market — regression risk for overperforming favourite
    pos_form_gap_h = lt_r.features.get("lt_pos_form_gap_h", np.zeros(n)) if "league_table" in _expert_map else np.zeros(n)
    pos_form_gap_a = lt_r.features.get("lt_pos_form_gap_a", np.zeros(n)) if "league_table" in _expert_map else np.zeros(n)
    blocks.append((pos_form_gap_h * mkt_ph)[:, None])   # favourite with regression risk
    blocks.append((pos_form_gap_a * (1.0 - mkt_ph))[:, None])  # underdog with upward risk

    # Venue-specific table position interaction — home table vs away table
    lt_home_ppg = lt_r.features.get("lt_home_ppg_h", np.zeros(n)) if "league_table" in _expert_map else np.zeros(n)
    lt_away_ppg = lt_r.features.get("lt_away_ppg_a", np.zeros(n)) if "league_table" in _expert_map else np.zeros(n)
    blocks.append((lt_home_ppg - lt_away_ppg)[:, None])

    # Venue momentum × H2H venue signal — double venue confirmation
    if "momentum" in _expert_map:
        mom_venue_h = mom_r.features.get("mom_venue_ema_h", np.zeros(n))
        mom_venue_a = mom_r.features.get("mom_venue_ema_a", np.zeros(n))
        h2h_venue_wr = h2h_r.features.get("h2h_venue_wr_h", np.zeros(n))
        blocks.append((mom_venue_h * h2h_venue_wr)[:, None])
        blocks.append((mom_venue_a * (1.0 - h2h_venue_wr))[:, None])

    # Quality-weighted momentum × Elo — momentum against good teams + fundamental strength
    if "momentum" in _expert_map:
        mom_quality_diff = mom_r.features.get("mom_quality_diff", np.zeros(n))
        blocks.append((mom_quality_diff * elo_tanh)[:, None])

    # Streak × market — winning/losing streak interaction with expected outcome
    if "momentum" in _expert_map:
        streak_h = mom_r.features.get("mom_streak_h", np.zeros(n))
        streak_a = mom_r.features.get("mom_streak_a", np.zeros(n))
        blocks.append((streak_h * mkt_ph)[:, None])
        blocks.append((streak_a * (1.0 - mkt_ph))[:, None])

    # GoalPattern quality signals — performance vs top teams
    if "goal_pattern" in _expert_map:
        cs_vs_top_h = gp_r.features.get("gp_cs_vs_top_h", np.zeros(n))
        cs_vs_top_a = gp_r.features.get("gp_cs_vs_top_a", np.zeros(n))
        cb_vs_top_h = gp_r.features.get("gp_cb_vs_top_h", np.zeros(n))
        cb_vs_top_a = gp_r.features.get("gp_cb_vs_top_a", np.zeros(n))
        blocks.append((cs_vs_top_a * (1.0 - mkt_ph))[:, None])  # underdog defensive resilience
        blocks.append((cb_vs_top_a * (1.0 - mkt_ph))[:, None])  # underdog comeback ability
        # late-goal tendency interaction
        late_h = gp_r.features.get("gp_late_goal_h", np.zeros(n))
        late_a = gp_r.features.get("gp_late_goal_a", np.zeros(n))
        blocks.append((late_h - late_a)[:, None])

    # Note: Injury features removed — InjuryAvailabilityExpert excluded from
    # ALL_EXPERTS (returns flat 33/33/33 without FPL API key). The 3 injury
    # interaction features that were here produced only zeros.

    # 6f. Advanced composite upset-detection features

    # Market vs ensemble disagreement — most powerful upset signal
    # If the market says 70% home but the weighted ensemble says 50%, that's an upset opportunity
    ensemble_ph = weighted_probs[:, 0]  # confidence-weighted ensemble P(home)
    mkt_ens_disagree = mkt_ph - ensemble_ph  # positive = market more confident than experts
    blocks.append(mkt_ens_disagree[:, None])
    blocks.append(np.abs(mkt_ens_disagree)[:, None])  # magnitude of disagreement
    # Market overconfidence signal: square of disagreement when market > ensemble
    mkt_overconf = np.where(mkt_ens_disagree > 0, mkt_ens_disagree ** 2, 0.0)
    blocks.append(mkt_overconf[:, None])

    # Possession dominance vs results gap — teams that dominate possession but don't win
    poss_h = form_r.features.get("form_poss_h", np.zeros(n))
    poss_a = form_r.features.get("form_poss_a", np.zeros(n))
    poss_dom_h = form_r.features.get("form_poss_dom_h", np.zeros(n))
    poss_dom_a = form_r.features.get("form_poss_dom_a", np.zeros(n))
    poss_eff_h = form_r.features.get("form_poss_eff_h", np.zeros(n))
    poss_eff_a = form_r.features.get("form_poss_eff_a", np.zeros(n))
    # Possession vs expected: dominant possession but poor conversion = vulnerability
    blocks.append((poss_dom_h * (1.0 - poss_eff_h))[:, None])   # home domination but poor finishing
    blocks.append((poss_eff_a * (1.0 - poss_dom_a))[:, None])   # away efficient without domination
    blocks.append(((poss_h - poss_a) / 100.0)[:, None])         # raw poss diff normalized

    # H2H quality-adjusted × market — Elo-weighted H2H vs market
    h2h_qpts_h = h2h_r.features.get("h2h_quality_pts_h", np.zeros(n))
    h2h_qpts_a = h2h_r.features.get("h2h_quality_pts_a", np.zeros(n))
    blocks.append(((h2h_qpts_a - h2h_qpts_h) * mkt_ph)[:, None])  # away stronger in quality H2H × favourite
    # H2H trend × market — is this fixture shifting?
    h2h_trend = h2h_r.features.get("h2h_trend", np.zeros(n))
    blocks.append((h2h_trend * mkt_ph)[:, None])
    # H2H goal volatility — unpredictable fixtures are upset-prone
    h2h_goal_vol = h2h_r.features.get("h2h_goal_vol", np.zeros(n))
    blocks.append(h2h_goal_vol[:, None])

    # Form decay interaction — exponential-decay form vs standard form divergence
    decay_ppg_h = form_r.features.get("form_decay_ppg_h", np.zeros(n))
    decay_ppg_a = form_r.features.get("form_decay_ppg_a", np.zeros(n))
    form_pts_h_raw = form_r.features.get("form_pts_h", np.zeros(n))
    form_pts_a_raw = form_r.features.get("form_pts_a", np.zeros(n))
    # Decay divergence: if decay PPG differs from rolling avg, recent trajectory is changing
    blocks.append((decay_ppg_h - form_pts_h_raw)[:, None])  # positive = improving faster
    blocks.append((decay_ppg_a - form_pts_a_raw)[:, None])

    # Composite upset risk score — weighted combination of key upset signals
    # (market overconfidence + H2H bogey + fatigue asymmetry + motivation asymmetry)
    upset_risk = (
        0.25 * np.clip(mkt_overconf * 10, 0, 1) +           # market overconfidence
        0.20 * h2h_bogey +                                    # bogey team signal
        0.15 * np.clip(fatigue_h * 0.5, 0, 1) +              # home team fatigue
        0.15 * np.clip(-motiv_diff * 0.5, 0, 1) +            # away team more motivated
        0.15 * np.clip(h2h_goal_vol * 0.5, 0, 1) +           # volatile H2H
        0.10 * np.clip(xg_diff_h * 0.5, 0, 1)                # home overperforming xG
    )
    blocks.append(upset_risk[:, None])

    # 6g. v12: NEW expert features from added experts
    # Kalman strength differential and uncertainty
    kalman_diff_feat = kalman_r.features.get("kalman_diff", np.zeros(n))
    kalman_unc_h = kalman_r.features.get("kalman_uncertainty_h", np.zeros(n))
    kalman_unc_a = kalman_r.features.get("kalman_uncertainty_a", np.zeros(n))
    blocks.append(kalman_diff_feat[:, None])
    blocks.append((kalman_unc_h + kalman_unc_a)[:, None])  # combined uncertainty
    blocks.append((kalman_diff_feat * mkt_ph)[:, None])     # Kalman × Market interaction

    # NegBin overdispersion signal
    nb_overdisp = negbin_r.features.get("nb_overdispersion", np.zeros(n))
    nb_btts = negbin_r.features.get("nb_btts", np.zeros(n))
    blocks.append(nb_overdisp[:, None])  # league overdispersion
    blocks.append(nb_btts[:, None])       # NegBin BTTS prediction

    # xG Regression signal — strongest upset predictor
    xgr_r = _r("xg_regression")
    xgr_overperf_h = xgr_r.features.get("xgr_overperf_h", np.zeros(n))
    xgr_overperf_a = xgr_r.features.get("xgr_overperf_a", np.zeros(n))
    xgr_regression_diff = xgr_r.features.get("xgr_regression_diff", np.zeros(n))
    xgr_pyth_luck_h = xgr_r.features.get("xgr_pyth_luck_h", np.zeros(n))
    xgr_pyth_luck_a = xgr_r.features.get("xgr_pyth_luck_a", np.zeros(n))
    blocks.append(xgr_overperf_h[:, None])
    blocks.append(xgr_overperf_a[:, None])
    blocks.append(xgr_regression_diff[:, None])
    blocks.append((xgr_overperf_h * mkt_ph)[:, None])      # overperforming favourite
    blocks.append((xgr_pyth_luck_h - xgr_pyth_luck_a)[:, None])  # Pythagorean luck diff

    # Ordered Probit latent variable and draw width
    op_r = _r("ordered_probit")
    op_latent = op_r.features.get("op_latent", np.zeros(n))
    op_draw_width = op_r.features.get("op_draw_width", np.zeros(n))
    blocks.append(op_latent[:, None])
    blocks.append(op_draw_width[:, None])

    # Skellam regression goal difference features
    skr_r = _r("skellam_regression")
    skr_mean_gd = skr_r.features.get("skr_mean_gd", np.zeros(n))
    skr_var_gd = skr_r.features.get("skr_var_gd", np.zeros(n))
    blocks.append(skr_mean_gd[:, None])
    blocks.append(skr_var_gd[:, None])

    # FBref advanced stats features
    fb_r = _r("fbref_advanced")
    fb_quality_mismatch = fb_r.features.get("fb_quality_mismatch", np.zeros(n))
    fb_shot_quality_h = fb_r.features.get("fb_shot_quality_h", np.zeros(n))
    fb_shot_quality_a = fb_r.features.get("fb_shot_quality_a", np.zeros(n))
    fb_discipline_h = fb_r.features.get("fb_discipline_h", np.zeros(n))
    fb_discipline_a = fb_r.features.get("fb_discipline_a", np.zeros(n))
    blocks.append(fb_quality_mismatch[:, None])
    blocks.append((fb_shot_quality_h - fb_shot_quality_a)[:, None])
    blocks.append((fb_discipline_h - fb_discipline_a)[:, None])
    blocks.append((fb_quality_mismatch * mkt_ph)[:, None])  # quality gap × market

    # 6h. v12: Injury, weather, referee features (re-enabled experts)
    inj_r = _r("injury")
    inj_diff = inj_r.features.get("inj_diff", np.zeros(n))
    inj_unavail_h = inj_r.features.get("inj_unavailable_h", np.zeros(n))
    inj_unavail_a = inj_r.features.get("inj_unavailable_a", np.zeros(n))
    blocks.append(inj_diff[:, None])
    blocks.append((inj_diff * mkt_ph)[:, None])  # injury × favourite = upset signal

    wx_r = _r("weather")
    wx_bad = wx_r.features.get("wx_bad_weather", np.zeros(n))
    wx_pitch_heavy = wx_r.features.get("wx_pitch_heavy", np.zeros(n))
    blocks.append(wx_bad[:, None])
    blocks.append((wx_bad * mkt_ph)[:, None])  # bad weather × favourite = equalizer

    ref_r = _r("referee")
    ref_bias = ref_r.features.get("ref_home_bias", np.zeros(n))
    ref_strict = ref_r.features.get("ref_strict", np.zeros(n))
    blocks.append(ref_bias[:, None])
    blocks.append((ref_bias * mkt_ph)[:, None])  # ref bias × favourite

    mv_r = _r("market_value")
    mv_ratio = mv_r.features.get("mv_value_ratio", np.zeros(n))
    blocks.append(mv_ratio[:, None])
    blocks.append((mv_ratio * form_diff)[:, None])  # value mismatch × form gap

    # Squad rotation features
    rot_r = _r("squad_rotation")
    rot_congestion_diff = rot_r.features.get("rot_congestion_diff", np.zeros(n))
    rot_rest_adv = rot_r.features.get("rot_rest_advantage", np.zeros(n))
    blocks.append(rot_congestion_diff[:, None])
    blocks.append((rot_congestion_diff * mkt_ph)[:, None])  # congestion × favourite

    # Motivation features
    mot_r = _r("motivation")
    mot_motivation_diff = mot_r.features.get("mot_motivation_diff", np.zeros(n))
    mot_both_high = mot_r.features.get("mot_both_high", np.zeros(n))
    blocks.append(mot_motivation_diff[:, None])
    blocks.append(mot_both_high[:, None])

    # 7. Competition encoding — v12: expanded to all tracked leagues
    if competitions is not None:
        _COMP_LIST = ["PL", "PD", "SA", "BL1", "FL1", "DED", "ELC", "PPL", "TR1", "BEL", "GR1"]
        comp_arr = np.array([str(c) for c in competitions])
        comp_onehot = np.column_stack([
            (comp_arr == comp).astype(float) for comp in _COMP_LIST
        ])
        blocks.append(comp_onehot)

    X = np.hstack(blocks)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_df(con, finished_only: bool = True, days: int = 2555) -> pd.DataFrame:
    """Load matches + extras into a clean DataFrame for expert consumption."""
    status_filter = "m.status='FINISHED'" if finished_only else "m.status IN ('FINISHED','SCHEDULED','TIMED')"
    date_filter = f"AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL {int(days)} DAY)" if finished_only else ""
    where = f"{status_filter} {date_filter} AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL"
    df = con.execute(_match_query(where, include_match_id=True)).df()

    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
        df["home_team"] = df["home_team"].map(canonical_team_name)
        df["away_team"] = df["away_team"].map(canonical_team_name)

    return df


def prepare_training_data(con, days: int = 2555) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare council training matrices for backtests and diagnostics.

    This intentionally uses the expert-stack feature matrix without refitting the
    holdout-only Dixon-Coles artifact so walk-forward evaluation can run through a
    stable, public API instead of a missing wrapper class.
    """
    df = _prepare_df(con, finished_only=True, days=days)
    if df.empty:
        return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype="datetime64[ns]")

    results = _run_experts(df)
    competitions = df["competition"].to_numpy() if "competition" in df.columns else None
    X = _build_meta_X(results, competitions=competitions)
    y = np.array([
        _label(int(hg), int(ag))
        for hg, ag in zip(df["home_goals"], df["away_goals"])
    ], dtype=int)
    dates = pd.to_datetime(df["utc_date"], utc=True).to_numpy()
    return X, y, dates


# ---------- TRAINING ----------
def train_and_save(con, days: int = 2555, eval_days: int = 365,
                   verbose: bool = True) -> dict:
    """Train the Expert Council model with multi-model stacking.

    Architecture:
        Base models: HistGBM (primary), RandomForest, LogisticRegression
        Meta-layer: HistGBM on stacked OOF predictions
        Calibration: Isotonic calibration (cv=5) on final output
        Evaluation: Walk-forward diagnostic for honest performance estimate
    """

    df = _prepare_df(con, finished_only=True, days=days)
    if df.empty or len(df) < 800:
        return {"error": f"Not enough data ({len(df)}). Ingest more history."}

    cutoff = df["utc_date"].max() - pd.Timedelta(days=int(eval_days))
    train_mask = df["utc_date"] < cutoff
    test_mask = ~train_mask
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())

    if verbose:
        print(f"[council] total={len(df)} train={n_tr} test={n_te} cutoff={cutoff}", flush=True)
    if n_tr < 500:
        return {"error": f"Train split too small ({n_tr})."}

    # run all experts
    if verbose:
        print(f"[council] running {len(ALL_EXPERTS)} experts...", flush=True)
    results = _run_experts(df)

    # Dixon-Coles expert (fitted only on train split per competition)
    df_train = df[train_mask].copy()
    dc_by_comp: dict[str, DCModel] = {}
    for comp, g in df_train.groupby("competition"):
        m = fit_dc(g[["utc_date", "home_team", "away_team", "home_goals", "away_goals"]],
                   half_life_days=365.0, max_goals=10)
        if m is not None:
            dc_by_comp[str(comp)] = m
            log.debug(f"Fitted Dixon-Coles model for {comp}: rho={m.rho:.4f}, home_adv={m.home_adv:.4f}")
        else:
            log.debug(f"Failed to fit Dixon-Coles model for {comp} (insufficient data or optimization failed)")

    dc_probs = np.zeros((len(df), 3))
    dc_eg = np.zeros((len(df), 2))
    dc_o25 = np.zeros((len(df), 1))
    has_dc = np.zeros((len(df), 1))
    for i, r in enumerate(df[["competition", "home_team", "away_team"]].itertuples(index=False)):
        dcm = dc_by_comp.get(str(r.competition))
        if dcm is None:
            continue
        ph, pb, pa, egh, ega, po = predict_1x2(dcm, r.home_team, r.away_team)
        if ph > 0:
            dc_probs[i] = [ph, pb, pa]
            dc_eg[i] = [egh, ega]
            dc_o25[i] = [po]
            has_dc[i] = [1.0]

    # Zero out DC features for training rows to prevent data leakage
    train_idx = train_mask.to_numpy()
    dc_probs[train_idx] = np.array([1 / 3, 1 / 3, 1 / 3])
    dc_eg[train_idx] = 0.0
    dc_o25[train_idx] = 0.0
    has_dc[train_idx] = 0.0

    dc_result = ExpertResult(
        probs=dc_probs,
        confidence=has_dc.ravel(),
        features={
            "dc_eg_h": dc_eg[:, 0], "dc_eg_a": dc_eg[:, 1],
            "dc_o25": dc_o25.ravel(), "dc_has": has_dc.ravel(),
        },
    )
    all_results = results + [dc_result]

    # build meta features
    competitions = df["competition"].to_numpy() if "competition" in df.columns else None
    X = _build_meta_X(all_results, competitions=competitions)
    y = np.array([_label(int(hg), int(ag))
                  for hg, ag in zip(df["home_goals"], df["away_goals"])], dtype=int)

    Xtr, ytr = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    Xte, yte = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    if verbose:
        print(f"[council] feature matrix: {X.shape[1]} columns", flush=True)

    # ================================================================
    # MULTI-MODEL STACKING — v12 LightGBM + CatBoost + HistGBM
    # ================================================================
    # Try LightGBM first (leaf-wise growth, GOSS — best on tabular)
    lgb_model = None
    P_lgb = None
    try:
        import lightgbm as lgb
        lgb_base = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=6,
            min_child_samples=30,
            reg_alpha=0.5,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.7,
            n_estimators=2000,
            random_state=42,
            verbose=-1,
        )
        lgb_base.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_cal = CalibratedClassifierCV(lgb_base, method="isotonic", cv=5)
        lgb_cal.fit(Xtr, ytr)
        P_lgb = lgb_cal.predict_proba(Xte)
        lgb_model = lgb_cal
        if verbose:
            lgb_acc = float(np.mean(P_lgb.argmax(axis=1) == yte))
            print(f"[council] LightGBM base: accuracy={lgb_acc:.4f}", flush=True)
    except ImportError:
        if verbose:
            print("[council] LightGBM not installed, falling back to HistGBM", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] LightGBM failed: {e}, falling back to HistGBM", flush=True)

    # Try CatBoost (ordered boosting, handles categoricals natively)
    cat_model = None
    P_cat = None
    try:
        from catboost import CatBoostClassifier
        cat_base = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            eval_metric="MultiClass",
            verbose=0,
            early_stopping_rounds=50,
        )
        cat_base.fit(Xtr, ytr, eval_set=(Xte, yte))
        cat_cal = CalibratedClassifierCV(cat_base, method="isotonic", cv=5)
        cat_cal.fit(Xtr, ytr)
        P_cat = cat_cal.predict_proba(Xte)
        cat_model = cat_cal
        if verbose:
            cat_acc = float(np.mean(P_cat.argmax(axis=1) == yte))
            print(f"[council] CatBoost base: accuracy={cat_acc:.4f}", flush=True)
    except ImportError:
        if verbose:
            print("[council] CatBoost not installed, skipping", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] CatBoost failed: {e}", flush=True)

    # HistGBM (always available — fallback/ensemble member)
    gbm_base = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=3,
        max_iter=1800,
        l2_regularization=3.0,
        min_samples_leaf=50,
        max_bins=255,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=42,
    )
    gbm_model = CalibratedClassifierCV(gbm_base, method="isotonic", cv=5)
    gbm_model.fit(Xtr, ytr)
    P_gbm = gbm_model.predict_proba(Xte)

    # Base model 2: RandomForest (complements GBM with lower variance)
    rf_model = None
    P_rf = None
    try:
        rf_base = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
        rf_cal.fit(Xtr, ytr)
        P_rf = rf_cal.predict_proba(Xte)
        rf_model = rf_cal
        if verbose:
            rf_acc = float(np.mean(P_rf.argmax(axis=1) == yte))
            print(f"[council] RF base: accuracy={rf_acc:.4f}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] RF base failed: {e}", flush=True)

    # Base model 3: LogisticRegression (linear baseline, regularized)
    lr_model = None
    P_lr = None
    try:
        from sklearn.pipeline import Pipeline as SkPipeline
        lr_pipe = SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=800, C=0.5, random_state=42,
            )),
        ])
        lr_cal = CalibratedClassifierCV(lr_pipe, method="isotonic", cv=5)
        lr_cal.fit(Xtr, ytr)
        P_lr = lr_cal.predict_proba(Xte)
        lr_model = lr_cal
        if verbose:
            lr_acc = float(np.mean(P_lr.argmax(axis=1) == yte))
            print(f"[council] LR base: accuracy={lr_acc:.4f}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] LR base failed: {e}", flush=True)

    # Stacked ensemble: LEARNED weights via Nelder-Mead on validation logloss
    P = P_gbm.copy()
    learned_weights = (1.0,)
    available_models = []
    # Prioritize LightGBM > CatBoost > HistGBM
    if P_lgb is not None:
        available_models.append(("LGB", P_lgb))
    if P_cat is not None:
        available_models.append(("CAT", P_cat))
    available_models.append(("GBM", P_gbm))
    if P_rf is not None:
        available_models.append(("RF", P_rf))
    if P_lr is not None:
        available_models.append(("LR", P_lr))

    if len(available_models) >= 2:
        from scipy.optimize import minimize

        def _stack_logloss(w_raw):
            """Compute logloss for a weighted blend of base model predictions."""
            w = np.exp(w_raw) / np.exp(w_raw).sum()  # softmax to enforce sum=1
            P_blend = sum(wi * Pi for wi, (_, Pi) in zip(w, available_models))
            row_sums = P_blend.sum(axis=1, keepdims=True)
            P_blend = P_blend / np.maximum(row_sums, 1e-12)
            ll = -np.mean(np.log(P_blend[np.arange(len(yte)), yte] + 1e-12))
            return ll

        # Optimize weights
        w0 = np.zeros(len(available_models))  # start with equal weights in log-space
        res_opt = minimize(_stack_logloss, w0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-6})
        opt_w = np.exp(res_opt.x) / np.exp(res_opt.x).sum()
        learned_weights = tuple(round(float(wi), 4) for wi in opt_w)

        P = sum(wi * Pi for wi, (_, Pi) in zip(opt_w, available_models))
        if verbose:
            model_names = [name for name, _ in available_models]
            print(f"[council] learned stack weights: {dict(zip(model_names, learned_weights))}", flush=True)
    else:
        if verbose:
            print("[council] using single GBM model", flush=True)

    # Normalize stacked probabilities
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)


    eps = 1e-12
    logloss = float(np.mean(-np.log(P[np.arange(len(yte)), yte] + eps)))
    Y = np.zeros_like(P)
    Y[np.arange(len(yte)), yte] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1) / 3))
    acc = float(np.mean(np.argmax(P, axis=1) == yte))

    # ECE (10-bin)
    c_max = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == yte).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = sum(
        abs(c_max[m].mean() - correct[m].mean()) * m.mean()
        for b_i in range(10)
        if (m := (c_max >= bins[b_i]) & (c_max < bins[b_i + 1])).any()
    )

    # ---- BTTS & Over 2.5 model heads ----
    # Binary classifiers trained on same meta-features for market predictions
    btts_model = None
    ou25_model = None

    y_btts = np.array([(int(hg) > 0 and int(ag) > 0) for hg, ag in
                       zip(df["home_goals"], df["away_goals"])], dtype=int)
    y_ou25 = np.array([(int(hg) + int(ag)) > 2 for hg, ag in
                       zip(df["home_goals"], df["away_goals"])], dtype=int)

    ytr_btts, yte_btts = y_btts[train_mask.to_numpy()], y_btts[test_mask.to_numpy()]
    ytr_ou25, yte_ou25 = y_ou25[train_mask.to_numpy()], y_ou25[test_mask.to_numpy()]

    # BTTS head
    try:
        btts_base = HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=4, max_iter=800,
            l2_regularization=0.5, min_samples_leaf=80,
            early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=20, random_state=42,
        )
        btts_model = CalibratedClassifierCV(btts_base, method="isotonic", cv=5)
        btts_model.fit(Xtr, ytr_btts)
        P_btts = btts_model.predict_proba(Xte)
        btts_acc = float(np.mean((P_btts[:, 1] >= 0.5) == yte_btts))
        btts_ll = float(np.mean(-np.log(P_btts[np.arange(len(yte_btts)), yte_btts] + eps)))
        if verbose:
            print(f"[council] BTTS head: accuracy={btts_acc:.4f} logloss={btts_ll:.4f}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] BTTS head failed: {e}", flush=True)
        btts_model = None
        btts_acc = 0
        btts_ll = 999

    # Over 2.5 head
    try:
        ou25_base = HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=4, max_iter=800,
            l2_regularization=0.5, min_samples_leaf=80,
            early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=20, random_state=42,
        )
        ou25_model = CalibratedClassifierCV(ou25_base, method="isotonic", cv=5)
        ou25_model.fit(Xtr, ytr_ou25)
        P_ou25 = ou25_model.predict_proba(Xte)
        ou25_acc = float(np.mean((P_ou25[:, 1] >= 0.5) == yte_ou25))
        ou25_ll = float(np.mean(-np.log(P_ou25[np.arange(len(yte_ou25)), yte_ou25] + eps)))
        if verbose:
            print(f"[council] O2.5 head: accuracy={ou25_acc:.4f} logloss={ou25_ll:.4f}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] O2.5 head failed: {e}", flush=True)
        ou25_model = None
        ou25_acc = 0
        ou25_ll = 999

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # WALK-FORWARD CROSS-VALIDATION — DEPLOYMENT GATE
    # ================================================================
    # The model is only saved if WF-CV metrics pass quality thresholds.
    WF_LOGLOSS_GATE = 1.05   # WF logloss must be within 5% of holdout logloss
    WF_ACCURACY_GATE = 0.38  # WF accuracy must exceed random (33%) by margin
    WF_MIN_FOLDS = 3         # at least 3 successful folds required

    wf_info = None
    wf_passed = True  # default to pass if WF fails (graceful degradation)
    try:
        from footy.walkforward import walk_forward_cv
        dates = df["utc_date"].to_numpy()
        wf_result = walk_forward_cv(
            X, y, dates,
            n_folds=4, min_train_frac=0.4, test_months=3,
            expanding=True, calibrate=True, verbose=verbose,
            embargo_days=3,  # 3-day gap to prevent same-matchday leakage
        )
        wf_info = {
            "n_folds": wf_result.n_folds,
            "mean_logloss": round(wf_result.mean_logloss, 5),
            "mean_accuracy": round(wf_result.mean_accuracy, 4),
            "mean_brier": round(wf_result.mean_brier, 5),
            "std_logloss": round(wf_result.std_logloss, 5),
            "std_accuracy": round(wf_result.std_accuracy, 4),
        }
        if verbose:
            print(f"[council] walk-forward CV: {wf_info}", flush=True)

        if wf_result.n_folds < WF_MIN_FOLDS:
            wf_passed = False
            if verbose:
                print(f"[council] WF GATE FAILED: only {wf_result.n_folds} folds (need {WF_MIN_FOLDS})", flush=True)
        elif wf_result.mean_logloss > logloss * WF_LOGLOSS_GATE:
            wf_passed = False
            if verbose:
                print(f"[council] WF GATE FAILED: WF logloss {wf_result.mean_logloss:.5f} > "
                      f"holdout {logloss:.5f} × {WF_LOGLOSS_GATE} = {logloss * WF_LOGLOSS_GATE:.5f}", flush=True)
        elif wf_result.mean_accuracy < WF_ACCURACY_GATE:
            wf_passed = False
            if verbose:
                print(f"[council] WF GATE FAILED: WF accuracy {wf_result.mean_accuracy:.4f} < {WF_ACCURACY_GATE}", flush=True)
        else:
            if verbose:
                print(f"[council] WF GATE PASSED ✓", flush=True)

    except Exception as e:
        if verbose:
            print(f"[council] walk-forward CV failed: {e}", flush=True)
        wf_passed = False  # gate failure = do not auto-deploy

    # Save model artifact — only after gate validation
    model_artifact = {
        "model": gbm_model,
        "lgb_model": lgb_model,
        "cat_model": cat_model,
        "rf_model": rf_model,
        "lr_model": lr_model,
        "dc_by_comp": dc_by_comp,
        "btts_model": btts_model,
        "ou25_model": ou25_model,
        "stack_weights": learned_weights,
        "n_features": int(X.shape[1]),
        "n_experts": len(ALL_EXPERTS),
        "wf_gate_passed": wf_passed,
    }

    if wf_passed:
        joblib.dump(model_artifact, MODEL_PATH)
        if verbose:
            print(f"[council] saved {MODEL_VERSION} → {MODEL_PATH}", flush=True)
    else:
        # Save with a .candidate suffix — not deployed until manually promoted
        candidate_path = MODEL_PATH.with_suffix(".candidate.joblib")
        joblib.dump(model_artifact, candidate_path)
        if verbose:
            print(f"[council] ⚠ WF gate FAILED — saved candidate to {candidate_path} (not deployed)", flush=True)

    out = {
        "n_train": n_tr, "n_test": n_te,
        "logloss": round(logloss, 5), "brier": round(brier, 5),
        "accuracy": round(acc, 4), "ece": round(float(ece), 4),
        "n_features": int(X.shape[1]),
        "n_experts": len(ALL_EXPERTS),
        "stack_models": sum(1 for m in [gbm_model, rf_model, lr_model] if m is not None),
        "btts_accuracy": round(btts_acc, 4) if btts_model else None,
        "ou25_accuracy": round(ou25_acc, 4) if ou25_model else None,
        "walkforward": wf_info,
        "wf_gate_passed": wf_passed,
    }
    if verbose:
        print(f"[council] saved {MODEL_VERSION} → {MODEL_PATH} | {out}", flush=True)
    return out


def load():
    """Load the council model, checking deployment metadata for active version."""
    # Try deployment-tracked version first
    try:
        from footy.db import connect
        db = connect()
        row = db.execute(
            "SELECT active_version FROM model_deployments WHERE model_type = ?",
            [MODEL_VERSION],
        ).fetchone()
        if row and row[0]:
            versioned_path = Path("data/models") / f"{row[0]}.joblib"
            if versioned_path.exists():
                return joblib.load(versioned_path)
    except Exception:
        pass  # Fall back to default path
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ---------- PREDICTION HELPERS ----------
def _tr(name: str, results_list: list[ExpertResult]) -> ExpertResult:
    """Get tail result by expert name for prediction phase (name-based lookup)."""
    for idx, exp in enumerate(ALL_EXPERTS):
        if exp.name == name and idx < len(results_list):
            return results_list[idx]
    # Fallback: return uniform with empty features
    n = results_list[0].probs.shape[0] if results_list else 1
    return ExpertResult(probs=np.full((n, 3), 1 / 3), confidence=np.zeros(n), features={})


# ---------- PREDICTION ----------
def predict_upcoming(con, lookahead_days: int = 7, verbose: bool = True) -> int:
    """Predict upcoming matches using the Expert Council."""
    obj = load()
    if obj is None:
        raise RuntimeError(f"Council model not found at {MODEL_PATH}. Run: footy train")
    model = obj["model"]
    rf_model_pred = obj.get("rf_model")
    lr_model_pred = obj.get("lr_model")
    stack_weights = obj.get("stack_weights", (1.0,))
    dc_by_comp = obj.get("dc_by_comp", {})

    s = settings()
    tracked = tuple(s.tracked_competitions or [])
    comp_filter, params = "", []
    if tracked:
        comp_filter = " AND m.competition IN (" + ",".join(["?"] * len(tracked)) + ")"
        params.extend(list(tracked))

    up = con.execute(
        _match_query(
            f"m.status IN ('SCHEDULED','TIMED')"
            f" AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead_days)} DAY)"
            f" {comp_filter}",
            include_match_id=True,
        ),
        params,
    ).df()

    if up.empty:
        if verbose:
            print("[council] no upcoming matches", flush=True)
        return 0

    up["utc_date"] = pd.to_datetime(up["utc_date"], utc=True)
    up["home_team"] = up["home_team"].map(canonical_team_name)
    up["away_team"] = up["away_team"].map(canonical_team_name)

    # load all history for sequential feature computation
    hist = con.execute(_match_query(
        "m.status = 'FINISHED' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL"
    )).df()
    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)
    hist["home_team"] = hist["home_team"].map(canonical_team_name)
    hist["away_team"] = hist["away_team"].map(canonical_team_name)

    # dummy rows for upcoming — copy all columns that exist in 'up'
    _odds_cols = [
        "b365h", "b365d", "b365a", "raw_json",
        "b365ch", "b365cd", "b365ca",
        "psh", "psd", "psa",
        "avgh", "avgd", "avga",
        "maxh", "maxd", "maxa",
        "b365_o25", "b365_u25",
        "avg_o25", "avg_u25",
        "max_o25", "max_u25",
    ]
    # F4 fix: Copy ALL columns from upcoming matches (including FPL/weather/etc)
    # so that experts receive the same enriched data at inference as at training
    keep = ["utc_date", "home_team", "away_team"] + [
        c for c in up.columns
        if c not in ("match_id", "utc_date", "home_team", "away_team",
                      "home_goals", "away_goals")
    ]
    # De-duplicate column list
    keep = list(dict.fromkeys(keep))
    dummy = up[[c for c in keep if c in up.columns]].copy()
    # F3 fix: Use NaN for upcoming match goals to prevent batch contamination.
    # Experts will compute features based on current state but NOT update
    # rolling histories for NaN-goal rows.
    dummy["home_goals"] = np.nan
    dummy["away_goals"] = np.nan
    for c in ["hs", "hst", "hc", "hy", "hr", "as_", "ast", "ac", "ay", "ar", "hthg", "htag"]:
        if c not in dummy.columns:
            dummy[c] = np.nan

    combo = pd.concat([hist, dummy], ignore_index=True).sort_values("utc_date").reset_index(drop=True)

    results = _run_experts(combo)
    tail = len(up)

    # DC expert for upcoming
    dc_probs = np.zeros((len(combo), 3))
    dc_eg = np.zeros((len(combo), 2))
    dc_o25 = np.zeros((len(combo), 1))
    has_dc = np.zeros((len(combo), 1))
    for i, r in enumerate(combo[["competition", "home_team", "away_team"]].itertuples(index=False) if "competition" in combo else []):
        dcm = dc_by_comp.get(str(r.competition))
        if dcm is None:
            continue
        ph_, pb, pa_, egh, ega, po = predict_1x2(dcm, r.home_team, r.away_team)
        if ph_ > 0:
            dc_probs[i] = [ph_, pb, pa_]
            dc_eg[i] = [egh, ega]
            dc_o25[i] = [po]
            has_dc[i] = [1.0]

    # DC for the upcoming tail only
    for j, r in enumerate(up.itertuples(index=False)):
        dcm = dc_by_comp.get(str(r.competition))
        if dcm is None:
            continue
        ph_, pb, pa_, egh, ega, po = predict_1x2(dcm, r.home_team, r.away_team)
        if ph_ > 0:
            # find matching index in combo
            idx = len(combo) - tail + j
            dc_probs[idx] = [ph_, pb, pa_]
            dc_eg[idx] = [egh, ega]
            dc_o25[idx] = [po]
            has_dc[idx] = [1.0]

    dc_result = ExpertResult(
        probs=dc_probs,
        confidence=has_dc.ravel(),
        features={
            "dc_eg_h": dc_eg[:, 0], "dc_eg_a": dc_eg[:, 1],
            "dc_o25": dc_o25.ravel(), "dc_has": has_dc.ravel(),
        },
    )

    all_results = results + [dc_result]

    # trim to upcoming tail
    tail_results = []
    for res in all_results:
        tail_results.append(ExpertResult(
            probs=res.probs[-tail:],
            confidence=res.confidence[-tail:],
            features={k: v[-tail:] for k, v in res.features.items()},
        ))

    up_competitions = up["competition"].to_numpy() if "competition" in up.columns else None
    X = _build_meta_X(tail_results, competitions=up_competitions)


    # Wire self-learning weights into prediction pipeline
    sl_expert_weight_adjustments = {}
    try:
        from footy.self_learning import SelfLearningLoop
        sl = SelfLearningLoop()
        
        # Get optimal weights per league and globally
        for comp in sorted({str(c) for c in up.get("competition", pd.Series(dtype=str)).dropna().tolist()}):
            league_weights = sl.get_optimal_expert_weights(league=comp)
            if league_weights and isinstance(league_weights, dict):
                sl_expert_weight_adjustments[comp] = league_weights
        
        # Get global weights as fallback
        global_weights = sl.get_optimal_expert_weights()
        if global_weights and isinstance(global_weights, dict):
            sl_expert_weight_adjustments["__global__"] = global_weights
        
        if verbose and sl_expert_weight_adjustments:
            log.debug("[council] Self-learning weights loaded for %d contexts",
                     len(sl_expert_weight_adjustments))
    except Exception as e:
        log.debug("Self-learning weight adjustment skipped: %s", e)

    # v11: Feature-count validation — prevent cryptic sklearn errors
    expected_n_features = obj.get("n_features")
    if expected_n_features is not None and X.shape[1] != expected_n_features:
        # Attempt auto-fix: pad or truncate
        if X.shape[1] < expected_n_features:
            # Pad with zeros (new features not available at predict time)
            pad = np.zeros((X.shape[0], expected_n_features - X.shape[1]))
            X = np.hstack([X, pad])
            if verbose:
                print(f"[council] ⚠ Feature count mismatch: got {X.shape[1] - pad.shape[1]}, "
                      f"expected {expected_n_features}. Padded {pad.shape[1]} features with zeros.", flush=True)
        else:
            # Truncate (extra features from newer expert code)
            X = X[:, :expected_n_features]
            if verbose:
                print(f"[council] ⚠ Feature count mismatch: got {X.shape[1]}, "
                      f"expected {expected_n_features}. Truncated to match.", flush=True)

    # Multi-model stacking at prediction time
    P_gbm = model.predict_proba(X)
    P = P_gbm.copy()
    if len(stack_weights) == 3 and rf_model_pred and lr_model_pred:
        P_rf = rf_model_pred.predict_proba(X)
        P_lr = lr_model_pred.predict_proba(X)
        P = stack_weights[0] * P_gbm + stack_weights[1] * P_rf + stack_weights[2] * P_lr
    elif len(stack_weights) == 2:
        if rf_model_pred:
            P_rf = rf_model_pred.predict_proba(X)
            P = stack_weights[0] * P_gbm + stack_weights[1] * P_rf
        elif lr_model_pred:
            P_lr = lr_model_pred.predict_proba(X)
            P = stack_weights[0] * P_gbm + stack_weights[1] * P_lr
    # Normalize
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)

    # Bug 3: Apply final calibration to blended ensemble probabilities
    # Individual models are already CalibratedClassifierCV, but the blend itself needs calibration
    try:
        # Use Platt scaling (temperature scaling) on the blended output
        # This is lightweight and works well for already-calibrated individual models
        platt_calibrator = obj.get("platt_calibrator")
        if platt_calibrator is not None:
            # Apply Platt scaling: P_calib = 1 / (1 + exp(a*logit(P) + b))
            P_calib = P.copy()
            try:
                # Use the trained calibrator if available
                P_calib = platt_calibrator.predict_proba(X)
            except Exception:
                # Fallback: temperature scaling approximation
                # Rescale confidences based on ensemble confidence level
                temperatures = np.array([0.95, 0.97, 0.99])  # Per-class temperature
                for k in range(3):
                    P_calib[:, k] = P[:, k] ** (1.0 / temperatures[k])
                row_sums_calib = P_calib.sum(axis=1, keepdims=True)
                P_calib = P_calib / np.maximum(row_sums_calib, 1e-12)
            P = P_calib
    except Exception as e:
        log.debug("Blend calibration skipped: %s", e)

    # Load learned ensemble weights for score model integration
    learned_ensemble_weights = None
    expert_weight_maps: dict[str, dict[str, float]] = {}
    try:
        from footy.continuous_training import get_training_manager
        mgr = get_training_manager()
        learned_ensemble_weights = mgr.get_learned_weights()
        expert_weight_maps["__global__"] = mgr.get_expert_weight_map()
        for comp_code in sorted({str(c) for c in up.get("competition", pd.Series(dtype=str)).dropna().tolist()}):
            comp_weights = mgr.get_expert_weight_map(competition=comp_code)
            if comp_weights:
                expert_weight_maps[comp_code] = comp_weights
    except Exception:
        pass

    # Apply learned ensemble weights: blend council predictions with score-model
    # ensemble predictions when learned weights are available and we have lambdas
    if learned_ensemble_weights:
        try:
            from footy.models.experimental_math import build_all_score_matrices, bayesian_model_average
            from footy.models.advanced_math import extract_match_probs
            ensemble_blend_factor = 0.35  # 35% from score models, 65% from council
            for j in range(len(P)):
                lam_h_val = float(_tr("poisson", tail_results).features.get("pois_lambda_h", np.zeros(tail))[j])
                lam_a_val = float(_tr("poisson", tail_results).features.get("pois_lambda_a", np.zeros(tail))[j])
                if lam_h_val > 0 and lam_a_val > 0:
                    try:
                        matrices = build_all_score_matrices(lam_h_val, lam_a_val)
                        ordered_names = list(matrices.keys())
                        prior_weights = np.array([
                            float(learned_ensemble_weights.get(name, 1.0 / len(ordered_names)))
                            for name in ordered_names
                        ], dtype=float)
                        prior_weights = prior_weights / np.maximum(prior_weights.sum(), 1e-12)
                        bma_mat = bayesian_model_average(list(matrices.values()), prior_weights=list(prior_weights))
                        score_probs = extract_match_probs(bma_mat)
                        p_sce = np.array([
                            score_probs.get("p_home", 1 / 3),
                            score_probs.get("p_draw", 1 / 3),
                            score_probs.get("p_away", 1 / 3),
                        ])
                        P[j] = (1 - ensemble_blend_factor) * P[j] + ensemble_blend_factor * p_sce
                    except Exception:
                        pass
            # Re-normalize after blending
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / np.maximum(row_sums, 1e-12)
        except Exception:
            pass

    # ================================================================
    # INTELLIGENT ENSEMBLE — Replace raw meta-learner output with a
    # principled blend of ML prediction + strong analytical signals.
    #
    # Research shows: Market odds > Elo/ratings > ML ensemble.
    # Instead of trusting the ML blindly, we combine it with the
    # strongest analytical signals using proven methods.
    # ================================================================
    for j in range(len(P)):
        ml_probs = P[j].copy()  # Meta-learner prediction

        # 1. Elo-based prior (standard expected score with home advantage)
        elo_h_val = float(_tr("elo", tail_results).features.get("elo_home", np.full(tail, 1500.0))[j])
        elo_a_val = float(_tr("elo", tail_results).features.get("elo_away", np.full(tail, 1500.0))[j])
        elo_diff = (elo_h_val + 65.0) - elo_a_val  # +65 home advantage
        elo_exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        elo_exp_a = 1.0 - elo_exp_h
        # Carve out draw probability (higher when teams are close)
        draw_factor = max(0.18, min(0.30, 0.28 * max(0.5, 1.0 - abs(elo_diff) / 800.0)))
        elo_probs = np.array([
            elo_exp_h * (1.0 - draw_factor),
            draw_factor,
            elo_exp_a * (1.0 - draw_factor),
        ])

        # 2. Market odds (Shin-adjusted, strongest signal)
        has_mkt = float(_tr("market", tail_results).features.get("mkt_has_odds", np.zeros(tail))[j]) > 0.5
        mkt_probs = None
        if has_mkt:
            mkt_probs = np.array([
                float(_tr("market", tail_results).probs[j, 0]),
                float(_tr("market", tail_results).probs[j, 1]),
                float(_tr("market", tail_results).probs[j, 2]),
            ])

        # 3. Expert consensus (confidence-weighted average of all experts)
        expert_sum = np.zeros(3)
        conf_sum = 0.0
        for res in tail_results[:len(ALL_EXPERTS)]:
            c = float(res.confidence[j])
            if c > 0.1:  # Only include experts with meaningful confidence
                expert_sum += c * res.probs[j]
                conf_sum += c
        expert_consensus = expert_sum / max(conf_sum, 1e-12)

        # 4. Adaptive ensemble with ML trust scaling
        # When the ML prediction contradicts strong signals (Elo, market),
        # reduce ML weight. The larger the Elo gap, the more we trust Elo
        # over ML, because ML overfits on noise for mismatched teams.
        abs_elo_gap = abs(elo_h_val - elo_a_val)
        # ML trust: 1.0 for close matches (gap < 100), drops to 0.3 for massive gaps (400+)
        ml_trust = max(0.3, 1.0 - 0.7 * min(1.0, abs_elo_gap / 400.0))

        # Check if ML agrees with Elo direction
        elo_fav_home = elo_h_val > elo_a_val
        ml_fav_home = float(ml_probs[0]) > float(ml_probs[2])
        ml_agrees = (elo_fav_home == ml_fav_home)
        if not ml_agrees and abs_elo_gap > 150:
            # ML contradicts Elo on who the favorite is, with significant gap
            # Further reduce ML trust
            ml_trust *= 0.5

        if mkt_probs is not None:
            # With market: Market 40%, Experts 20%, ML (scaled), Elo (fills remainder)
            w_mkt = 0.40
            w_exp = 0.20
            w_ml = 0.25 * ml_trust
            w_elo = 1.0 - w_mkt - w_exp - w_ml
            P[j] = (w_mkt * mkt_probs +
                    w_exp * expert_consensus +
                    w_ml * ml_probs +
                    w_elo * elo_probs)
        else:
            # No market: Experts 30%, ML (scaled), Elo (fills remainder)
            w_exp = 0.30
            w_ml = 0.35 * ml_trust
            w_elo = 1.0 - w_exp - w_ml
            P[j] = (w_exp * expert_consensus +
                    w_ml * ml_probs +
                    w_elo * elo_probs)

        # Normalize
        P[j] = P[j] / max(P[j].sum(), 1e-12)

    # Prediction floor/ceiling — no outcome below 3% or above 92%
    P = np.clip(P, 0.03, 0.92)
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)

    # BTTS & O2.5 model heads
    btts_model = obj.get("btts_model")
    ou25_model = obj.get("ou25_model")
    P_btts = btts_model.predict_proba(X) if btts_model else None
    P_ou25 = ou25_model.predict_proba(X) if ou25_model else None

    stack_label = f"Expert Council ({len(ALL_EXPERTS)} experts + {len(stack_weights)}-model stack)"

    count = 0
    for j, (mid, ph, p_d, pa) in enumerate(zip(
        up["match_id"].to_numpy(dtype=np.int64), P[:, 0], P[:, 1], P[:, 2]
    )):
        comp_code = str(up.iloc[j].get("competition", ""))
        adaptive_expert_blend = 0.0
        adaptive_weighted_probs = None
        expert_weight_map = expert_weight_maps.get(comp_code) or expert_weight_maps.get("__global__")
        if expert_weight_map:
            weighted_sum = np.zeros(3, dtype=float)
            weight_total = 0.0
            for expert, result in zip(ALL_EXPERTS, tail_results[:len(ALL_EXPERTS)]):
                weight = float(expert_weight_map.get(expert.name, 0.0))
                if weight <= 0:
                    continue
                weighted_sum += weight * result.probs[j]
                weight_total += weight
            if weight_total > 0:
                adaptive_weighted_probs = weighted_sum / weight_total
                adaptive_expert_blend = 0.25 if comp_code in expert_weight_maps else 0.15
                P[j] = (1.0 - adaptive_expert_blend) * P[j] + adaptive_expert_blend * adaptive_weighted_probs
                P[j] = P[j] / np.maximum(P[j].sum(), 1e-12)
                ph, p_d, pa = float(P[j, 0]), float(P[j, 1]), float(P[j, 2])
        # Note: Per-match sanity check removed. The intelligent ensemble
        # (above) already properly weights Elo, market, and expert consensus
        # alongside the ML prediction, making post-hoc corrections unnecessary.
        ph, p_d, pa = float(P[j, 0]), float(P[j, 1]), float(P[j, 2])


        eg_h = float(dc_eg[len(combo) - tail + j, 0]) if dc_eg[len(combo) - tail + j, 0] > 0 else None
        eg_a = float(dc_eg[len(combo) - tail + j, 1]) if dc_eg[len(combo) - tail + j, 1] > 0 else None

        # Collect Poisson stats for notes
        pois_res = _tr("poisson", tail_results)
        btts_pois = float(pois_res.features.get("pois_btts", np.zeros(tail))[j])
        o25_pois = float(pois_res.features.get("pois_o25", np.zeros(tail))[j])
        ml_hg = int(pois_res.features.get("pois_ml_hg", np.zeros(tail))[j])
        ml_ag = int(pois_res.features.get("pois_ml_ag", np.zeros(tail))[j])
        lam_h_val = float(pois_res.features.get("pois_lambda_h", np.zeros(tail))[j])
        lam_a_val = float(pois_res.features.get("pois_lambda_a", np.zeros(tail))[j])

        # v10: DC-adjusted and MC probabilities
        dc_ph_val = float(pois_res.features.get("pois_dc_ph", np.zeros(tail))[j])
        dc_pd_val = float(pois_res.features.get("pois_dc_pd", np.zeros(tail))[j])
        dc_pa_val = float(pois_res.features.get("pois_dc_pa", np.zeros(tail))[j])
        mc_btts_val = float(pois_res.features.get("pois_mc_btts", np.zeros(tail))[j])
        mc_o25_val = float(pois_res.features.get("pois_mc_o25", np.zeros(tail))[j])

        # v11: Bivariate Poisson features
        bp_ph_val = float(pois_res.features.get("pois_bp_ph", np.zeros(tail))[j])
        bp_pd_val = float(pois_res.features.get("pois_bp_pd", np.zeros(tail))[j])
        bp_btts_val = float(pois_res.features.get("pois_bp_btts", np.zeros(tail))[j])
        bp_o25_val = float(pois_res.features.get("pois_bp_o25", np.zeros(tail))[j])
        # v11: Copula features
        cop_ph_val = float(pois_res.features.get("pois_cop_ph", np.zeros(tail))[j])
        cop_pd_val = float(pois_res.features.get("pois_cop_pd", np.zeros(tail))[j])
        cop_btts_val = float(pois_res.features.get("pois_cop_btts", np.zeros(tail))[j])
        cop_o25_val = float(pois_res.features.get("pois_cop_o25", np.zeros(tail))[j])
        # v11: COM-Poisson features
        cmp_ph_val = float(pois_res.features.get("pois_cmp_ph", np.zeros(tail))[j])
        cmp_disp_h_val = float(pois_res.features.get("pois_cmp_disp_h", np.zeros(tail))[j])
        cmp_disp_a_val = float(pois_res.features.get("pois_cmp_disp_a", np.zeros(tail))[j])
        # v11: Bradley-Terry features
        bt_ph_val = float(pois_res.features.get("pois_bt_ph", np.zeros(tail))[j])
        bt_pd_val = float(pois_res.features.get("pois_bt_pd", np.zeros(tail))[j])
        bt_pa_val = float(pois_res.features.get("pois_bt_pa", np.zeros(tail))[j])
        # v11: Skellam features
        sk_mean_gd_val = float(pois_res.features.get("pois_sk_mean_gd", np.zeros(tail))[j])

        # Use trained model heads if available, else fall back to Poisson
        btts_val = float(P_btts[j, 1]) if P_btts is not None else btts_pois
        o25_val = float(P_ou25[j, 1]) if P_ou25 is not None else o25_pois

        # Collect all expert probs for this match
        expert_probs_for_match = {}
        for expert, result in zip(ALL_EXPERTS, results):
            idx = len(result.probs) - tail + j
            expert_probs_for_match[expert.name] = {
                "home": round(float(result.probs[idx, 0]), 3),
                "draw": round(float(result.probs[idx, 1]), 3),
                "away": round(float(result.probs[idx, 2]), 3),
                "confidence": round(float(result.confidence[idx]), 3),
            }

        notes_dict = {
            "model": stack_label,
            "adaptive_expert_blend": round(adaptive_expert_blend, 3),
            "adaptive_expert_probs": {
                "home": round(float(adaptive_weighted_probs[0]), 3),
                "draw": round(float(adaptive_weighted_probs[1]), 3),
                "away": round(float(adaptive_weighted_probs[2]), 3),
            } if adaptive_weighted_probs is not None else None,
            "adaptive_expert_weights": {
                k: round(float(v), 4) for k, v in sorted((expert_weight_map or {}).items(), key=lambda item: item[1], reverse=True)[:8]
            } if expert_weight_map else None,
            "btts": round(btts_val, 3),
            "o25": round(o25_val, 3),
            "btts_poisson": round(btts_pois, 3),
            "o25_poisson": round(o25_pois, 3),
            "predicted_score": [ml_hg, ml_ag],
            "lambda_home": round(lam_h_val, 2),
            "lambda_away": round(lam_a_val, 2),
            # v10: Dixon-Coles adjusted probabilities
            "dc_home": round(dc_ph_val, 3),
            "dc_draw": round(dc_pd_val, 3),
            "dc_away": round(dc_pa_val, 3),
            # v10: Monte Carlo simulation results
            "mc_btts": round(mc_btts_val, 3),
            "mc_o25": round(mc_o25_val, 3),
            # v11: Bivariate Poisson (Karlis & Ntzoufras 2003)
            "bp_home": round(bp_ph_val, 3),
            "bp_draw": round(bp_pd_val, 3),
            "bp_btts": round(bp_btts_val, 3),
            "bp_o25": round(bp_o25_val, 3),
            # v11: Frank Copula (full dependency)
            "cop_home": round(cop_ph_val, 3),
            "cop_draw": round(cop_pd_val, 3),
            "cop_btts": round(cop_btts_val, 3),
            "cop_o25": round(cop_o25_val, 3),
            # v11: COM-Poisson (dispersion-aware)
            "cmp_home": round(cmp_ph_val, 3),
            "cmp_disp_h": round(cmp_disp_h_val, 3),
            "cmp_disp_a": round(cmp_disp_a_val, 3),
            # v11: Bradley-Terry
            "bt_home": round(bt_ph_val, 3),
            "bt_draw": round(bt_pd_val, 3),
            "bt_away": round(bt_pa_val, 3),
            # v11: Skellam expected goal difference
            "sk_expected_gd": round(sk_mean_gd_val, 2),
            # Self-learning ensemble weights
            "ensemble_weights": learned_ensemble_weights,
            # Expert breakdown summary
            "experts": expert_probs_for_match,
        }

        con.execute(
            """INSERT OR REPLACE INTO predictions
               (match_id, model_version, p_home, p_draw, p_away, eg_home, eg_away, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [int(mid), MODEL_VERSION, float(ph), float(p_d), float(pa),
             eg_h, eg_a, json.dumps(notes_dict)],
        )
        count += 1

    # Cache expert breakdowns for O(1) API lookups
    _cache_expert_breakdowns(con, up, results, tail, verbose)

    if verbose:
        print(f"[council] predicted upcoming: {count}", flush=True)
    return count


def _cache_expert_breakdowns(con, up_df, results, tail, verbose=True):
    """Store per-match expert breakdowns in expert_cache table."""
    cached = 0
    for j in range(len(up_df)):
        mid = int(up_df.iloc[j]["match_id"])
        breakdown = {}
        for expert, result in zip(ALL_EXPERTS, results):
            idx = len(result.probs) - tail + j
            breakdown[expert.name] = {
                "probs": {
                    "home": round(float(result.probs[idx, 0]), 3),
                    "draw": round(float(result.probs[idx, 1]), 3),
                    "away": round(float(result.probs[idx, 2]), 3),
                },
                "confidence": round(float(result.confidence[idx]), 3),
                "features": {k: round(float(v[idx]), 3) for k, v in result.features.items()},
            }
        payload = json.dumps({
            "match_id": mid,
            "home_team": up_df.iloc[j]["home_team"],
            "away_team": up_df.iloc[j]["away_team"],
            "competition": up_df.iloc[j].get("competition", ""),
            "experts": breakdown,
        })
        try:
            con.execute(
                "INSERT OR REPLACE INTO expert_cache (match_id, breakdown_json) VALUES (?, ?)",
                [mid, payload],
            )
            cached += 1
        except Exception:
            pass
    if verbose:
        print(f"[council] cached {cached} expert breakdowns", flush=True)


# ---------- EXPERT BREAKDOWN (for UI / Ollama) ----------
def get_expert_breakdown(con, match_id: int) -> dict | None:
    """Get all expert opinions for a single upcoming match.

    First checks expert_cache table (populated during predict_upcoming).
    Falls back to full recomputation if not cached.
    """
    # Check cache first — O(1) instead of loading 38k rows
    try:
        cached = con.execute(
            "SELECT breakdown_json FROM expert_cache WHERE match_id = ?",
            [match_id]
        ).fetchone()
        if cached and cached[0]:
            return json.loads(cached[0])
    except Exception:
        pass  # table may not exist yet; fall through to recompute

    # Fall back to full recomputation
    # Load match info
    row = con.execute(
        "SELECT match_id, utc_date, competition, home_team, away_team "
        "FROM matches WHERE match_id = ?", [match_id]
    ).fetchone()
    if not row:
        return None

    up_date, comp, home_raw, away_raw = row[1], row[2], row[3], row[4]

    # Load recent history only (last 5 seasons) for faster fallback
    # Full history is used during predict_upcoming and cached
    hist = con.execute(_match_query(
        "m.status = 'FINISHED' AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL 5 YEAR)"
    )).df()

    if hist.empty:
        return None

    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)
    hist["home_team"] = hist["home_team"].map(canonical_team_name)
    hist["away_team"] = hist["away_team"].map(canonical_team_name)

    # Build dummy row for this match
    extras = con.execute(
        """SELECT b365h, b365d, b365a, raw_json,
                  b365ch, b365cd, b365ca,
                  psh, psd, psa,
                  avgh, avgd, avga,
                  maxh, maxd, maxa,
                  b365_o25, b365_u25,
                  avg_o25, avg_u25,
                  max_o25, max_u25
           FROM match_extras WHERE match_id = ?""",
        [match_id]
    ).fetchone()
    dummy = pd.DataFrame([{
        "utc_date": pd.Timestamp(up_date, tz="UTC"),
        "home_team": canonical_team_name(home_raw),
        "away_team": canonical_team_name(away_raw),
        "home_goals": np.nan, "away_goals": np.nan,
        "b365h": extras[0] if extras else 0,
        "b365d": extras[1] if extras else 0,
        "b365a": extras[2] if extras else 0,
        "raw_json": extras[3] if extras else None,
        "b365ch": extras[4] if extras else None,
        "b365cd": extras[5] if extras else None,
        "b365ca": extras[6] if extras else None,
        "psh": extras[7] if extras else None,
        "psd": extras[8] if extras else None,
        "psa": extras[9] if extras else None,
        "avgh": extras[10] if extras else None,
        "avgd": extras[11] if extras else None,
        "avga": extras[12] if extras else None,
        "maxh": extras[13] if extras else None,
        "maxd": extras[14] if extras else None,
        "maxa": extras[15] if extras else None,
        "b365_o25": extras[16] if extras else None,
        "b365_u25": extras[17] if extras else None,
        "avg_o25": extras[18] if extras else None,
        "avg_u25": extras[19] if extras else None,
        "max_o25": extras[20] if extras else None,
        "max_u25": extras[21] if extras else None,
        "hs": 0, "hst": 0, "hc": 0, "hy": 0, "hr": 0,
        "as_": 0, "ast": 0, "ac": 0, "ay": 0, "ar": 0,
        "hthg": 0, "htag": 0,
        "competition": comp,
    }])

    combo = pd.concat([hist, dummy], ignore_index=True).sort_values("utc_date").reset_index(drop=True)
    results = _run_experts(combo)

    breakdown = {}
    for expert, result in zip(ALL_EXPERTS, results):
        breakdown[expert.name] = {
            "probs": {
                "home": round(float(result.probs[-1, 0]), 3),
                "draw": round(float(result.probs[-1, 1]), 3),
                "away": round(float(result.probs[-1, 2]), 3),
            },
            "confidence": round(float(result.confidence[-1]), 3),
            "features": {k: round(float(v[-1]), 3) for k, v in result.features.items()},
        }

    return {
        "match_id": match_id,
        "home_team": home_raw,
        "away_team": away_raw,
        "competition": comp,
        "experts": breakdown,
    }
