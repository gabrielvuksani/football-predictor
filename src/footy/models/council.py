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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

MODEL_VERSION = "v13_oracle"
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
        gp_r.features.get("gp_cs_vs_top_h", np.zeros(n))
        cs_vs_top_a = gp_r.features.get("gp_cs_vs_top_a", np.zeros(n))
        gp_r.features.get("gp_cb_vs_top_h", np.zeros(n))
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
    inj_r.features.get("inj_unavailable_h", np.zeros(n))
    inj_r.features.get("inj_unavailable_a", np.zeros(n))
    blocks.append(inj_diff[:, None])
    blocks.append((inj_diff * mkt_ph)[:, None])  # injury × favourite = upset signal

    wx_r = _r("weather")
    wx_bad = wx_r.features.get("wx_bad_weather", np.zeros(n))
    wx_r.features.get("wx_pitch_heavy", np.zeros(n))
    blocks.append(wx_bad[:, None])
    blocks.append((wx_bad * mkt_ph)[:, None])  # bad weather × favourite = equalizer

    ref_r = _r("referee")
    ref_bias = ref_r.features.get("ref_home_bias", np.zeros(n))
    ref_r.features.get("ref_strict", np.zeros(n))
    blocks.append(ref_bias[:, None])
    blocks.append((ref_bias * mkt_ph)[:, None])  # ref bias × favourite

    mv_r = _r("market_value")
    mv_ratio = mv_r.features.get("mv_value_ratio", np.zeros(n))
    blocks.append(mv_ratio[:, None])
    blocks.append((mv_ratio * form_diff)[:, None])  # value mismatch × form gap

    # Squad rotation features
    rot_r = _r("squad_rotation")
    rot_congestion_diff = rot_r.features.get("rot_congestion_diff", np.zeros(n))
    rot_r.features.get("rot_rest_advantage", np.zeros(n))
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


# ---------------------------------------------------------------------------
# v13 Focused Feature Builder — ~55 features instead of 400+
# Research shows 15-25 well-chosen features outperform hundreds when SNR is low.
# This builder extracts the highest-signal features organized by tier.
# ---------------------------------------------------------------------------


def _build_v13_features(results: list[ExpertResult], experts: list[Expert] | None = None,
                        competitions: np.ndarray | None = None) -> np.ndarray:
    """Build focused feature matrix for v13 Oracle model.

    ~55 features organized by signal strength, replacing the 400+ feature matrix.
    Research (Groll 2024, Fischer & Heuer 2024) shows that feature selection
    matters more than model complexity for football prediction.
    """
    if experts is None:
        experts = ALL_EXPERTS
    n = results[0].probs.shape[0]
    features: dict[str, np.ndarray] = {}

    # Build name-based expert lookup
    _expert_map: dict[str, int] = {}
    for idx, exp in enumerate(experts):
        _expert_map[exp.name] = idx

    def _r(name: str) -> ExpertResult:
        idx = _expert_map.get(name)
        if idx is not None and idx < len(results):
            return results[idx]
        return ExpertResult(probs=np.full((n, 3), 1 / 3), confidence=np.zeros(n), features={})

    # Shorthand access to key experts
    elo_r = _r("elo"); mkt_r = _r("market"); form_r = _r("form")
    pois_r = _r("poisson"); h2h_r = _r("h2h"); ctx_r = _r("context")
    lt_r = _r("league_table"); mom_r = _r("momentum"); mot_r = _r("motivation")
    kalman_r = _r("kalman_elo"); glicko_r = _r("glicko2")
    xgr_r = _r("xg_regression"); rot_r = _r("squad_rotation")
    gp_r = _r("goal_pattern"); ref_r = _r("referee"); wx_r = _r("weather")
    inj_r = _r("injury"); mv_r = _r("market_value")

    # ── TIER 1: Strongest signal (~15 features) ──
    # Market implied probabilities (strongest single predictor per research)
    features["mkt_ph"] = mkt_r.probs[:, 0]
    features["mkt_pd"] = mkt_r.probs[:, 1]
    features["mkt_pa"] = mkt_r.probs[:, 2]

    # Elo ratings
    features["elo_diff"] = elo_r.features.get("elo_diff", np.zeros(n))
    features["elo_home_adv"] = elo_r.features.get("elo_home_adv", np.zeros(n))

    # Poisson expected goals
    features["pois_lambda_h"] = pois_r.features.get("pois_lambda_h", np.zeros(n))
    features["pois_lambda_a"] = pois_r.features.get("pois_lambda_a", np.zeros(n))

    # Dixon-Coles probabilities (from the last result if DC was appended)
    if len(results) > len(experts):
        dc_r = results[-1]  # DC result appended at end
        features["dc_ph"] = dc_r.probs[:, 0]
        features["dc_pd"] = dc_r.probs[:, 1]
        features["dc_pa"] = dc_r.probs[:, 2]
    else:
        features["dc_ph"] = pois_r.features.get("pois_dc_ph", np.zeros(n))
        features["dc_pd"] = pois_r.features.get("pois_dc_pd", np.zeros(n))
        features["dc_pa"] = pois_r.features.get("pois_dc_pa", np.zeros(n))

    # xG rolling
    features["xg_diff_h"] = form_r.features.get("form_xg_diff_h", np.zeros(n))
    features["xg_diff_a"] = form_r.features.get("form_xg_diff_a", np.zeros(n))

    # Form PPG rolling
    features["form_pts_h"] = form_r.features.get("form_pts_h", np.zeros(n))
    features["form_pts_a"] = form_r.features.get("form_pts_a", np.zeros(n))

    # Glicko-2 rating diff
    features["glicko_diff"] = glicko_r.features.get("glicko2_rating_diff", np.zeros(n))

    # ── TIER 2: Strong signal (~15 features) ──
    # Poisson-derived markets
    features["pois_btts"] = pois_r.features.get("pois_btts", np.zeros(n))
    features["pois_o25"] = pois_r.features.get("pois_o25", np.zeros(n))
    features["pois_cs00"] = pois_r.features.get("pois_cs00", np.zeros(n))

    # Market line movement
    features["mkt_move_h"] = mkt_r.features.get("mkt_move_h", np.zeros(n))
    features["mkt_move_a"] = mkt_r.features.get("mkt_move_a", np.zeros(n))

    # xG overperformance (Pythagorean luck — strongest regression signal)
    features["xgr_overperf_h"] = xgr_r.features.get("xgr_overperf_h", np.zeros(n))
    features["xgr_overperf_a"] = xgr_r.features.get("xgr_overperf_a", np.zeros(n))

    # H2H signals
    features["h2h_bogey"] = h2h_r.features.get("h2h_bogey", np.zeros(n))
    features["h2h_venue_wr"] = h2h_r.features.get("h2h_venue_wr_h", np.zeros(n))

    # Motivation & fatigue
    features["mot_diff"] = mot_r.features.get("mot_motivation_diff", np.zeros(n))
    features["ctx_high_stakes"] = ctx_r.features.get("ctx_high_stakes", np.zeros(n))
    features["ctx_fatigue_diff"] = ctx_r.features.get("ctx_fatigue_diff", np.zeros(n))
    features["rot_rest_adv"] = rot_r.features.get("rot_rest_advantage", np.zeros(n))

    # Table position
    features["lt_pos_diff"] = lt_r.features.get("lt_pos_diff", np.zeros(n))

    # Market value
    features["mv_ratio"] = mv_r.features.get("mv_value_ratio", np.zeros(n))

    # ── TIER 3: Moderate signal (~15 features) ──
    # Expert consensus stats
    all_probs = np.stack([r.probs for r in results], axis=0)
    mean_probs = np.nan_to_num(np.mean(all_probs, axis=0), nan=1.0/3)
    features["consensus_ph"] = mean_probs[:, 0]
    features["consensus_var_h"] = np.var(all_probs[:, :, 0], axis=0)
    features["consensus_entropy"] = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

    # Key expert agreements
    features["agree_elo_mkt"] = 1.0 - np.abs(elo_r.probs[:, 0] - mkt_r.probs[:, 0])
    features["agree_form_h2h"] = 1.0 - np.abs(form_r.probs[:, 0] - h2h_r.probs[:, 0])
    features["agree_pois_mkt"] = 1.0 - np.abs(pois_r.probs[:, 0] - mkt_r.probs[:, 0])

    # Copula (Frank) probability
    cop_r = _r("copula")
    features["cop_ph"] = cop_r.probs[:, 0]

    # Momentum
    features["mom_cross_diff"] = mom_r.features.get("mom_ema_cross_diff", np.zeros(n))

    # Goal patterns
    gp_first_h = gp_r.features.get("gp_first_goal_h", np.zeros(n))
    gp_first_a = gp_r.features.get("gp_first_goal_a", np.zeros(n))
    features["gp_first_goal_diff"] = gp_first_h - gp_first_a
    gp_cs_h = gp_r.features.get("gp_cs_vs_top_h", np.zeros(n))
    gp_cs_a = gp_r.features.get("gp_cs_vs_top_a", np.zeros(n))
    features["gp_cs_diff"] = gp_cs_h - gp_cs_a

    # Weather, referee, injury
    features["wx_bad"] = wx_r.features.get("wx_bad_weather", np.zeros(n))
    features["ref_bias"] = ref_r.features.get("ref_home_bias", np.zeros(n))
    features["inj_diff"] = inj_r.features.get("inj_diff", np.zeros(n))

    # ── TIER 4: Context + v13 New Expert Signals (~20 features) ──
    # Kalman strength
    features["kalman_diff"] = kalman_r.features.get("kalman_diff", np.zeros(n))

    # Upset composite: market-ensemble disagreement
    conf_weights = np.stack([r.confidence for r in results], axis=0)
    conf_sum = conf_weights.sum(axis=0, keepdims=True) + 1e-12
    weighted_ph = np.sum(all_probs[:, :, 0] * (conf_weights / conf_sum), axis=0)
    features["mkt_ens_disagree"] = mkt_r.probs[:, 0] - weighted_ph

    # Season progress
    sp_r = _r("seasonal_pattern")
    features["season_progress"] = sp_r.features.get("sp_season_progress", np.zeros(n))

    # Derby flag
    features["is_derby"] = ctx_r.features.get("ctx_is_derby", np.zeros(n))

    # Relegation flag
    features["is_relegation"] = ctx_r.features.get("ctx_is_relegation_a", np.zeros(n))

    # ── v13 NEW EXPERT FEATURES ──
    # Pythagorean expectation — strongest regression-to-mean signal
    pyth_r = _r("pythagorean")
    features["pyth_luck_h"] = pyth_r.features.get("pyth_luck_h", np.zeros(n))
    features["pyth_luck_a"] = pyth_r.features.get("pyth_luck_a", np.zeros(n))
    features["pyth_regression"] = pyth_r.features.get("pyth_regression_signal", np.zeros(n))

    # GAS dynamic strength — score-driven, robust to outliers
    gas_r = _r("gas")
    features["gas_diff"] = gas_r.features.get("gas_diff", np.zeros(n))
    features["gas_vol_h"] = gas_r.features.get("gas_volatility_h", np.zeros(n))
    features["gas_vol_a"] = gas_r.features.get("gas_volatility_a", np.zeros(n))

    # Adaptive Bayesian — spike-and-slab regime detection
    abs_r = _r("adaptive_bayesian")
    features["abs_diff"] = abs_r.features.get("abs_diff", np.zeros(n))
    features["abs_regime_h"] = abs_r.features.get("abs_regime_h", np.zeros(n))
    features["abs_regime_a"] = abs_r.features.get("abs_regime_a", np.zeros(n))

    # HMM team states — performance regime detection
    hmm_r = _r("hmm")
    features["hmm_dominant_h"] = hmm_r.features.get("hmm_dominant_h", np.zeros(n))
    features["hmm_dominant_a"] = hmm_r.features.get("hmm_dominant_a", np.zeros(n))
    features["hmm_vulnerable_h"] = hmm_r.features.get("hmm_vulnerable_h", np.zeros(n))
    features["hmm_vulnerable_a"] = hmm_r.features.get("hmm_vulnerable_a", np.zeros(n))

    # Transfer impact
    tf_r = _r("transfer")
    features["tf_value_dom"] = tf_r.features.get("tf_value_dominance", np.zeros(n))
    features["tf_depth_adv"] = tf_r.features.get("tf_depth_advantage", np.zeros(n))

    # News sentiment — media disruption detection
    ns_r = _r("news_sentiment")
    features["ns_tone_diff"] = ns_r.features.get("ns_tone_diff", np.zeros(n))
    features["ns_disruption_h"] = ns_r.features.get("ns_disruption_h", np.zeros(n))
    features["ns_disruption_a"] = ns_r.features.get("ns_disruption_a", np.zeros(n))

    # Betting movement — smart money detection
    bm_r = _r("betting_movement")
    features["bm_move_h"] = bm_r.features.get("bm_move_h", np.zeros(n))
    features["bm_move_a"] = bm_r.features.get("bm_move_a", np.zeros(n))
    features["bm_sharp_signal"] = bm_r.features.get("bm_sharp_signal", np.zeros(n))

    # Manager impact — coaching change detection
    mgr_r = _r("manager")
    features["mgr_bounce_h"] = mgr_r.features.get("mgr_bounce_h", np.zeros(n))
    features["mgr_bounce_a"] = mgr_r.features.get("mgr_bounce_a", np.zeros(n))
    features["mgr_stability_diff"] = (
        mgr_r.features.get("mgr_stability_h", np.zeros(n)) -
        mgr_r.features.get("mgr_stability_a", np.zeros(n))
    )

    # Lineup strength — rotation and availability
    lu_r = _r("lineup")
    features["lu_squad_diff"] = (
        lu_r.features.get("lu_squad_strength_h", np.zeros(n)) -
        lu_r.features.get("lu_squad_strength_a", np.zeros(n))
    )
    features["lu_rotation_h"] = lu_r.features.get("lu_rotation_risk_h", np.zeros(n))
    features["lu_rotation_a"] = lu_r.features.get("lu_rotation_risk_a", np.zeros(n))

    # ── UPSET INTERACTION FEATURES ──
    # These cross signals from multiple experts to detect upsets
    mkt_ph = features["mkt_ph"]
    # Pythagorean luck × market favourite — overperforming favourite regresses
    features["upset_pyth_mkt"] = features["pyth_luck_h"] * mkt_ph
    # HMM vulnerable × market favourite — regime mismatch
    features["upset_hmm_mkt"] = features["hmm_vulnerable_h"] * mkt_ph
    # Adaptive Bayesian regime change × market — team in flux
    features["upset_regime_mkt"] = features["abs_regime_h"] * mkt_ph
    # GAS volatility × fatigue — unpredictable + tired
    features["upset_vol_fatigue"] = features["gas_vol_h"] * features["ctx_fatigue_diff"]
    # News disruption × market favourite — bad news for favourite
    features["upset_news_mkt"] = features.get("ns_disruption_h", np.zeros(n)) * mkt_ph
    # Betting movement against favourite — sharp money opposing
    features["upset_sharp_mkt"] = features.get("bm_sharp_signal", np.zeros(n)) * (1.0 - mkt_ph)
    # Manager bounce for underdog — new manager energy
    features["upset_bounce_away"] = features.get("mgr_bounce_a", np.zeros(n)) * (1.0 - mkt_ph)
    # Lineup rotation for favourite — weakened team
    features["upset_rotation_mkt"] = features.get("lu_rotation_h", np.zeros(n)) * mkt_ph

    # ── MATCH DYNAMICS FEATURES ──
    md_r = _r("match_dynamics")
    features["md_comeback_diff"] = md_r.features.get("md_comeback_a", np.zeros(n)) - md_r.features.get("md_comeback_h", np.zeros(n))
    features["md_collapse_h"] = md_r.features.get("md_collapse_h", np.zeros(n))
    features["md_resilience_diff"] = md_r.features.get("md_resilience_a", np.zeros(n)) - md_r.features.get("md_resilience_h", np.zeros(n))
    features["md_streak_h"] = md_r.features.get("md_streak_h", np.zeros(n))
    features["md_streak_a"] = md_r.features.get("md_streak_a", np.zeros(n))
    features["md_streak_break_h"] = md_r.features.get("md_streak_break_h", np.zeros(n))
    features["md_card_diff"] = md_r.features.get("md_card_rate_h", np.zeros(n)) - md_r.features.get("md_card_rate_a", np.zeros(n))

    # ── SCHEDULE CONTEXT FEATURES ──
    sched_r = _r("schedule_context")
    features["sc_fatigue_diff"] = sched_r.features.get("sc_midweek_fatigue_h", np.zeros(n)) - sched_r.features.get("sc_midweek_fatigue_a", np.zeros(n))
    features["sc_post_intl"] = sched_r.features.get("sc_post_intl_break", np.zeros(n))
    features["sc_honeymoon_diff"] = sched_r.features.get("sc_promoted_honeymoon_h", np.zeros(n)) - sched_r.features.get("sc_promoted_honeymoon_a", np.zeros(n))
    features["sc_congestion"] = sched_r.features.get("sc_holiday_congestion", np.zeros(n))
    features["sc_schedule_adv"] = sched_r.features.get("sc_schedule_advantage", np.zeros(n))

    # ── ADDITIONAL UPSET INTERACTIONS with dynamics/schedule ──
    features["upset_streak_mkt"] = features["md_streak_break_h"] * mkt_ph  # winning streak about to break
    features["upset_fatigue_sched"] = features.get("sc_fatigue_diff", np.zeros(n)) * mkt_ph  # fatigued favourite
    features["upset_resilience_mkt"] = features["md_resilience_diff"] * (1.0 - mkt_ph)  # resilient underdog

    # ── OPPONENT-ADJUSTED FEATURES (research: top 3 feature category) ──
    oa_r = _r("opponent_adjusted")
    features["oa_strength_diff"] = oa_r.features.get("oa_strength_diff", np.zeros(n))
    features["oa_ppg_h"] = oa_r.features.get("oa_ppg_h", np.zeros(n))
    features["oa_ppg_a"] = oa_r.features.get("oa_ppg_a", np.zeros(n))
    features["ppda_diff"] = oa_r.features.get("ppda_h", np.zeros(n)) - oa_r.features.get("ppda_a", np.zeros(n))
    features["style_sp_diff"] = oa_r.features.get("style_set_piece_h", np.zeros(n)) - oa_r.features.get("style_set_piece_a", np.zeros(n))
    features["pts_safety_h"] = oa_r.features.get("pts_to_safety_h", np.zeros(n))
    features["pts_safety_a"] = oa_r.features.get("pts_to_safety_a", np.zeros(n))
    features["pts_title_h"] = oa_r.features.get("pts_to_title_h", np.zeros(n))
    features["post_intl_diff"] = oa_r.features.get("post_intl_record_h", np.zeros(n)) - oa_r.features.get("post_intl_record_a", np.zeros(n))

    # ── ADVANCED DERIVED FEATURES — Data manipulation for maximum signal ──
    # These combine multiple expert signals in ways proven by research to be predictive

    # 1. Elo × Form agreement — when rating AND form agree, prediction is stronger
    features["elo_form_agree"] = features["elo_diff"] * (features["form_pts_h"] - features["form_pts_a"])

    # 2. Market confidence — how extreme is the market favourite?
    features["mkt_confidence"] = np.abs(features["mkt_ph"] - features["mkt_pa"])

    # 3. Multi-model disagreement — when models diverge, upsets are more likely
    model_stack = np.column_stack([
        features["mkt_ph"], features.get("dc_ph", np.zeros(n)),
        features["consensus_ph"], features["cop_ph"],
        elo_r.probs[:, 0], pois_r.probs[:, 0],
    ])
    features["model_disagreement"] = np.std(model_stack, axis=1)

    # 4. Defensive quality matchup — strong defense vs weak attack = low scoring
    features["def_quality_mismatch"] = (
        form_r.features.get("form_ga_h", np.zeros(n)) -  # home goals conceded
        form_r.features.get("form_gf_a", np.zeros(n))     # away goals scored
    )

    # 5. Expected goals surplus — teams consistently outscoring xG will regress
    features["xg_surplus_h"] = features["form_pts_h"] - features.get("xgr_overperf_h", np.zeros(n))

    # 6. Combined upset risk — weighted sum of all upset signals
    upset_signals = np.column_stack([
        features.get("upset_pyth_mkt", np.zeros(n)),
        features.get("upset_hmm_mkt", np.zeros(n)),
        features.get("upset_regime_mkt", np.zeros(n)),
        features.get("upset_vol_fatigue", np.zeros(n)),
        features.get("upset_news_mkt", np.zeros(n)),
        features.get("upset_sharp_mkt", np.zeros(n)),
        features.get("upset_bounce_away", np.zeros(n)),
        features.get("upset_rotation_mkt", np.zeros(n)),
        features.get("upset_streak_mkt", np.zeros(n)),
        features.get("upset_fatigue_sched", np.zeros(n)),
        features.get("upset_resilience_mkt", np.zeros(n)),
    ])
    features["upset_composite"] = np.mean(upset_signals, axis=1)
    features["upset_max_signal"] = np.max(upset_signals, axis=1)
    features["upset_n_active"] = np.sum(upset_signals > 0.05, axis=1).astype(float)

    # Competition encoding — one-hot for top leagues (works with all sklearn models)
    if competitions is not None:
        _COMP_LIST = ["PL", "PD", "SA", "BL1", "FL1", "DED", "ELC", "PPL", "TR1", "BEL", "GR1"]
        comp_arr = np.array([str(c) for c in competitions])
        for comp_code in _COMP_LIST:
            features[f"comp_{comp_code}"] = (comp_arr == comp_code).astype(float)

    # Build numpy array from all numeric features
    numeric_keys = [k for k, v in features.items() if np.issubdtype(v.dtype, np.number)]
    blocks = [features[k][:, None] if features[k].ndim == 1 else features[k] for k in numeric_keys]

    X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


# Feature names for v13 (used for SHAP analysis and debugging)
V13_FEATURE_NAMES = [
    # Tier 1: Strongest signal
    "mkt_ph", "mkt_pd", "mkt_pa", "elo_diff", "elo_home_adv",
    "pois_lambda_h", "pois_lambda_a", "dc_ph", "dc_pd", "dc_pa",
    "xg_diff_h", "xg_diff_a", "form_pts_h", "form_pts_a", "glicko_diff",
    # Tier 2: Strong signal
    "pois_btts", "pois_o25", "pois_cs00", "mkt_move_h", "mkt_move_a",
    "xgr_overperf_h", "xgr_overperf_a", "h2h_bogey", "h2h_venue_wr",
    "mot_diff", "ctx_high_stakes", "ctx_fatigue_diff", "rot_rest_adv",
    "lt_pos_diff", "mv_ratio",
    # Tier 3: Moderate signal
    "consensus_ph", "consensus_var_h", "consensus_entropy",
    "agree_elo_mkt", "agree_form_h2h", "agree_pois_mkt", "cop_ph",
    "mom_cross_diff", "gp_first_goal_diff", "gp_cs_diff",
    "wx_bad", "ref_bias", "inj_diff",
    # Tier 4: Context + v13 new experts
    "kalman_diff", "mkt_ens_disagree", "season_progress",
    "is_derby", "is_relegation",
    # v13 new expert features
    "pyth_luck_h", "pyth_luck_a", "pyth_regression",
    "gas_diff", "gas_vol_h", "gas_vol_a",
    "abs_diff", "abs_regime_h", "abs_regime_a",
    "hmm_dominant_h", "hmm_dominant_a", "hmm_vulnerable_h", "hmm_vulnerable_a",
    "tf_value_dom", "tf_depth_adv",
    # News, betting, manager, lineup features
    "ns_tone_diff", "ns_disruption_h", "ns_disruption_a",
    "bm_move_h", "bm_move_a", "bm_sharp_signal",
    "mgr_bounce_h", "mgr_bounce_a", "mgr_stability_diff",
    "lu_squad_diff", "lu_rotation_h", "lu_rotation_a",
    # Upset interaction features
    "upset_pyth_mkt", "upset_hmm_mkt", "upset_regime_mkt", "upset_vol_fatigue",
    "upset_news_mkt", "upset_sharp_mkt", "upset_bounce_away", "upset_rotation_mkt",
    # Match dynamics features
    "md_comeback_diff", "md_collapse_h", "md_resilience_diff",
    "md_streak_h", "md_streak_a", "md_streak_break_h", "md_card_diff",
    # Schedule context features
    "sc_fatigue_diff", "sc_post_intl", "sc_honeymoon_diff", "sc_congestion", "sc_schedule_adv",
    # Additional upset interactions
    "upset_streak_mkt", "upset_fatigue_sched", "upset_resilience_mkt",
    # Advanced derived features
    "elo_form_agree", "mkt_confidence", "model_disagreement",
    "def_quality_mismatch", "xg_surplus_h",
    "upset_composite", "upset_max_signal", "upset_n_active",
    # Opponent-adjusted features
    "oa_strength_diff", "oa_ppg_h", "oa_ppg_a", "ppda_diff", "style_sp_diff",
    "pts_safety_h", "pts_safety_a", "pts_title_h", "post_intl_diff",
]


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
    """Train the Expert Council model — v13 Oracle architecture.

    Architecture:
        Layer 1: 34+ experts → ~55 focused features (v13) or 400+ full features (v12 compat)
        Layer 2: CatBoost primary (native categoricals, ordered boosting) + HistGBM fallback
        Layer 3: Temperature scaling calibration (single parameter T)
        Evaluation: Walk-forward cross-validation with temporal splits
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

    # build meta features — v13 focused features (research: 55 > 400+ when SNR is low)
    competitions = df["competition"].to_numpy() if "competition" in df.columns else None
    X = _build_v13_features(all_results, competitions=competitions)
    y = np.array([_label(int(hg), int(ag))
                  for hg, ag in zip(df["home_goals"], df["away_goals"])], dtype=int)

    Xtr, ytr = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    Xte, yte = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    if verbose:
        print(f"[council] v13 feature matrix: {X.shape[1]} columns (focused selection)", flush=True)

    # ================================================================
    # v13 ORACLE MODEL STACK — CatBoost primary + HistGBM fallback
    # Research: CatBoost > LightGBM for medium football datasets (ordered
    # boosting prevents prediction shift, native categoricals, better calibrated)
    # ================================================================

    # Primary: CatBoost (best for medium-size football data)
    cat_model = None
    P_cat = None
    try:
        from catboost import CatBoostClassifier
        cat_base = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.7,
            min_data_in_leaf=30,
            use_best_model=True,
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        )
        cat_base.fit(Xtr, ytr, eval_set=(Xte, yte))
        P_cat = cat_base.predict_proba(Xte)
        cat_model = cat_base
        if verbose:
            cat_acc = float(np.mean(P_cat.argmax(axis=1) == yte))
            print(f"[council] CatBoost primary: accuracy={cat_acc:.4f}", flush=True)
    except ImportError:
        if verbose:
            print("[council] CatBoost not installed, using HistGBM fallback", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] CatBoost failed: {e}, using HistGBM fallback", flush=True)

    # XGBoost (multi:softprob for calibrated 3-class probabilities)
    xgb_model = None
    P_xgb = None
    try:
        import xgboost as xgb
        xgb_base = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=5,
            min_child_weight=10,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.7,
            learning_rate=0.03,
            n_estimators=2000,
            reg_alpha=0.1,
            reg_lambda=3.0,
            max_delta_step=1,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        xgb_base.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        P_xgb = xgb_base.predict_proba(Xte)
        xgb_model = xgb_base
        if verbose:
            xgb_acc = float(np.mean(P_xgb.argmax(axis=1) == yte))
            print(f"[council] XGBoost: accuracy={xgb_acc:.4f}", flush=True)
    except ImportError:
        if verbose:
            print("[council] XGBoost not installed, skipping", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] XGBoost failed: {e}", flush=True)

    # Fallback: HistGBM (always available, no extra dependency)
    gbm_base = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=5,
        max_iter=2000,
        l2_regularization=3.0,
        min_samples_leaf=30,
        max_bins=255,
        max_leaf_nodes=63,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=50,
        random_state=42,
    )
    gbm_base.fit(Xtr, ytr)
    P_gbm = gbm_base.predict_proba(Xte)
    gbm_model = gbm_base

    # Baseline: LogisticRegression (interpretable, naturally calibrated)
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
        lr_pipe.fit(Xtr, ytr)
        P_lr = lr_pipe.predict_proba(Xte)
        lr_model = lr_pipe
        if verbose:
            lr_acc = float(np.mean(P_lr.argmax(axis=1) == yte))
            print(f"[council] LR baseline: accuracy={lr_acc:.4f}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[council] LR baseline failed: {e}", flush=True)

    # Select best primary model and learn stack weights
    available_models = []
    if P_cat is not None:
        available_models.append(("CAT", P_cat))
    if P_xgb is not None:
        available_models.append(("XGB", P_xgb))
    available_models.append(("GBM", P_gbm))
    if P_lr is not None:
        available_models.append(("LR", P_lr))

    # Stack: learned weights via Nelder-Mead on validation logloss
    P = P_cat if P_cat is not None else P_gbm
    learned_weights = (1.0,)

    if len(available_models) >= 2:
        from scipy.optimize import minimize

        def _stack_logloss(w_raw):
            w = np.exp(w_raw) / np.exp(w_raw).sum()
            P_blend = sum(wi * Pi for wi, (_, Pi) in zip(w, available_models))
            row_sums = P_blend.sum(axis=1, keepdims=True)
            P_blend = P_blend / np.maximum(row_sums, 1e-12)
            return -np.mean(np.log(P_blend[np.arange(len(yte)), yte] + 1e-12))

        w0 = np.zeros(len(available_models))
        res_opt = minimize(_stack_logloss, w0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-6})
        opt_w = np.exp(res_opt.x) / np.exp(res_opt.x).sum()
        learned_weights = tuple(round(float(wi), 4) for wi in opt_w)

        P = sum(wi * Pi for wi, (_, Pi) in zip(opt_w, available_models))
        if verbose:
            model_names = [name for name, _ in available_models]
            print(f"[council] stack weights: {dict(zip(model_names, learned_weights))}", flush=True)

    # Normalize stacked probabilities
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)

    # ── Temperature scaling — replaces isotonic calibration ──
    # Single parameter T that preserves discrimination while fixing calibration.
    # Naturally handles multiclass, probabilities always sum to 1.
    temperature = 1.0
    try:
        from scipy.special import softmax as scipy_softmax
        logits = np.log(P + 1e-12)

        def _temp_nll(T):
            T_val = float(T[0]) if hasattr(T, '__len__') else float(T)
            scaled = logits / max(T_val, 0.01)
            probs = scipy_softmax(scaled, axis=1)
            return -np.mean(np.log(probs[np.arange(len(yte)), yte] + 1e-12))

        from scipy.optimize import minimize as opt_minimize
        res_temp = opt_minimize(_temp_nll, x0=[1.5], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        temperature = float(res_temp.x[0])
        if verbose:
            print(f"[council] temperature scaling: T={temperature:.4f}", flush=True)

        # Apply temperature scaling to final predictions
        P = scipy_softmax(logits / temperature, axis=1)
    except Exception as e:
        if verbose:
            print(f"[council] temperature scaling failed: {e}, using raw probabilities", flush=True)


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
                print("[council] WF GATE PASSED ✓", flush=True)

    except Exception as e:
        if verbose:
            print(f"[council] walk-forward CV failed: {e}", flush=True)
        wf_passed = False  # gate failure = do not auto-deploy

    # Save model artifact — only after gate validation
    model_artifact = {
        "model": gbm_model,
        "cat_model": cat_model,
        "xgb_model": xgb_model,
        "lr_model": lr_model,
        "dc_by_comp": dc_by_comp,
        "btts_model": btts_model,
        "ou25_model": ou25_model,
        "stack_weights": learned_weights,
        "temperature": temperature,
        "n_features": int(X.shape[1]),
        "n_experts": len(ALL_EXPERTS),
        "wf_gate_passed": wf_passed,
        "version": "v13_oracle",
        "feature_names": V13_FEATURE_NAMES[:X.shape[1]],
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
        "stack_models": sum(1 for m in [cat_model, gbm_model, lr_model] if m is not None),
        "temperature": round(temperature, 4),
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
    """Predict upcoming matches using the v13 Oracle pipeline."""
    obj = load()
    if obj is None:
        raise RuntimeError(f"Council model not found at {MODEL_PATH}. Run: footy train")
    model = obj["model"]
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

    # ================================================================
    # v13 Clean Prediction Pipeline — 3 layers, no heuristic overrides
    #
    # The ML meta-learner already sees market odds, Elo, expert consensus
    # as INPUT FEATURES and learns the optimal combination from data.
    # No hard-coded weights needed. Trust the trained model.
    # ================================================================
    up_competitions = up["competition"].to_numpy() if "competition" in up.columns else None
    X = _build_v13_features(tail_results, competitions=up_competitions)

    # Feature-count validation
    expected_n_features = obj.get("n_features")
    if expected_n_features is not None and X.shape[1] != expected_n_features:
        if X.shape[1] < expected_n_features:
            pad = np.zeros((X.shape[0], expected_n_features - X.shape[1]))
            X = np.hstack([X, pad])
        else:
            X = X[:, :expected_n_features]

    # Layer 2: Multi-model stacking at prediction time
    cat_model_pred = obj.get("cat_model")
    xgb_model_pred = obj.get("xgb_model")
    lr_model_pred = obj.get("lr_model")

    available_preds = []
    if cat_model_pred is not None:
        try:
            available_preds.append(("CAT", cat_model_pred.predict_proba(X)))
        except Exception:
            pass
    if xgb_model_pred is not None:
        try:
            available_preds.append(("XGB", xgb_model_pred.predict_proba(X)))
        except Exception:
            pass
    available_preds.append(("GBM", model.predict_proba(X)))
    if lr_model_pred is not None:
        try:
            available_preds.append(("LR", lr_model_pred.predict_proba(X)))
        except Exception:
            pass

    # Apply learned stack weights
    if len(available_preds) >= 2 and len(stack_weights) == len(available_preds):
        P = sum(w * p for w, (_, p) in zip(stack_weights, available_preds))
    else:
        P = available_preds[0][1]

    # Normalize
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)

    # Layer 3: Temperature scaling (learned during training)
    temperature = obj.get("temperature", 1.0)
    if temperature != 1.0:
        try:
            from scipy.special import softmax as scipy_softmax
            logits = np.log(P + 1e-12)
            P = scipy_softmax(logits / temperature, axis=1)
        except Exception:
            pass

    # Soft floor/ceiling — gentler than hard clip, preserves relative ordering
    P = np.clip(P, 0.02, 0.95)
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-12)

    # BTTS & O2.5 model heads
    btts_model = obj.get("btts_model")
    ou25_model = obj.get("ou25_model")
    P_btts = btts_model.predict_proba(X) if btts_model else None
    P_ou25 = ou25_model.predict_proba(X) if ou25_model else None

    stack_label = f"v13 Oracle ({len(ALL_EXPERTS)} experts, T={temperature:.2f})"

    count = 0
    for j, (mid, ph, p_d, pa) in enumerate(zip(
        up["match_id"].to_numpy(dtype=np.int64), P[:, 0], P[:, 1], P[:, 2]
    )):
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
            # v13: stack weights
            "stack_weights": list(stack_weights),
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
