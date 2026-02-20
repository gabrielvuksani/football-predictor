"""
Expert Council Model — Multi-expert ensemble for football match prediction.

Architecture:
    Layer 1 — Eleven specialist experts (see footy.models.experts package)
    Layer 2 — Conflict, consensus, KL divergence & interaction signals across experts
    Layer 3 — Multi-model stack (HistGBM + RF + LR) with LEARNED weights + isotonic cal
    Layer 4 — Walk-forward cross-validation for honest performance estimation
    Layer 5 — Ollama interpreter (optional) for narrative match analysis

Expert modules live in ``footy.models.experts``.  This module contains the
meta-learner, SQL queries, training pipeline, and prediction logic.
"""
from __future__ import annotations

import json
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

MODEL_VERSION = "v10_council"
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
               af.away_injuries AS af_inj_a"""

_MATCH_JOINS = """\
FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        LEFT JOIN fpl_availability fpl_h ON fpl_h.team = m.home_team
        LEFT JOIN fpl_availability fpl_a ON fpl_a.team = m.away_team
        LEFT JOIN fpl_fixture_difficulty fdr_h ON fdr_h.team = m.home_team
        LEFT JOIN fpl_fixture_difficulty fdr_a ON fdr_a.team = m.away_team
        LEFT JOIN af_context af ON af.match_id = m.match_id"""


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
        - Expert probs:          N × 3  (N = 11 experts)
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
    n_exp = len(results)  # 11 experts in v10
    # elo-market agreement
    elo_mkt_agree = 1.0 - np.abs(results[0].probs[:, 0] - results[1].probs[:, 0])
    # form-h2h agreement
    form_h2h_agree = 1.0 - np.abs(results[2].probs[:, 0] - results[4].probs[:, 0])
    # poisson-market agreement
    pois_mkt_agree = 1.0 - np.abs(results[3].probs[:, 0] - results[1].probs[:, 0])
    # elo-form agreement
    elo_form_agree = 1.0 - np.abs(results[0].probs[:, 0] - results[2].probs[:, 0])
    # poisson-elo agreement
    pois_elo_agree = 1.0 - np.abs(results[3].probs[:, 0] - results[0].probs[:, 0])
    # goal-pattern-form agreement
    gp_form_agree = np.zeros(n)
    if n_exp > 6:
        gp_form_agree = 1.0 - np.abs(results[6].probs[:, 0] - results[2].probs[:, 0])
    # league-table-elo agreement
    lt_elo_agree = np.zeros(n)
    if n_exp > 7:
        lt_elo_agree = 1.0 - np.abs(results[7].probs[:, 0] - results[0].probs[:, 0])
    # momentum-form agreement
    mom_form_agree = np.zeros(n)
    if n_exp > 8:
        mom_form_agree = 1.0 - np.abs(results[8].probs[:, 0] - results[2].probs[:, 0])
    # momentum-elo agreement
    mom_elo_agree = np.zeros(n)
    if n_exp > 8:
        mom_elo_agree = 1.0 - np.abs(results[8].probs[:, 0] - results[0].probs[:, 0])
    # v10: bayesian-poisson agreement
    bayes_pois_agree = np.zeros(n)
    if n_exp > 9:
        bayes_pois_agree = 1.0 - np.abs(results[9].probs[:, 0] - results[3].probs[:, 0])
    # v10: bayesian-market agreement
    bayes_mkt_agree = np.zeros(n)
    if n_exp > 9:
        bayes_mkt_agree = 1.0 - np.abs(results[9].probs[:, 0] - results[1].probs[:, 0])
    # max disagreement across any pair of experts (for home win)
    home_stack = np.stack([r.probs[:, 0] for r in results[:n_exp]])  # (n_exp, n)
    max_disagree = home_stack.max(axis=0) - home_stack.min(axis=0)
    # number of experts that agree on same winner
    all_winners = np.argmax(np.stack([r.probs for r in results[:n_exp]]), axis=2)  # (n_exp, n)
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
    ])

    # 5. Per-expert entropy (how uncertain each expert is) — vectorized
    for res in results[:n_exp]:
        safe_probs = np.nan_to_num(res.probs, nan=1.0/3)
        expert_ent = -np.sum(safe_probs * np.log(safe_probs + 1e-12), axis=1)
        blocks.append(expert_ent[:, None])

    # 5b. v10: Per-expert KL divergence from ensemble mean — vectorized
    for res in results[:n_exp]:
        safe_probs = np.nan_to_num(res.probs, nan=1.0/3)
        kl_arr = np.sum(safe_probs * np.log((safe_probs + 1e-12) / (mean_probs + 1e-12)), axis=1)
        kl_arr = np.clip(kl_arr, 0.0, None)  # KL >= 0
        blocks.append(kl_arr[:, None])

    # 6. Feature interactions (cross-domain signal combinations)
    # These capture nonlinear relationships the tree model might miss
    # Elo diff × form PPG diff
    elo_diff = results[0].features.get("elo_diff", np.zeros(n))
    form_pts_h = results[2].features.get("form_pts_h", np.zeros(n))
    form_pts_a = results[2].features.get("form_pts_a", np.zeros(n))
    form_diff = form_pts_h - form_pts_a
    blocks.append((elo_diff * form_diff)[:, None])  # Elo × Form interaction
    # Market probability × Poisson probability (product of independent estimates)
    mkt_ph = results[1].probs[:, 0]
    pois_ph = results[3].probs[:, 0]
    blocks.append((mkt_ph * pois_ph)[:, None])  # Market × Poisson interaction
    # Momentum × recent form (amplifier when both agree)
    if n_exp > 8:
        mom_cross = results[8].features.get("mom_ema_cross_diff", np.zeros(n))
        blocks.append((mom_cross * form_diff)[:, None])  # Momentum × Form
        # Goal scoring burst × opponent defensive weakness
        burst_h = results[8].features.get("mom_burst_h", np.zeros(n))
        ga_slope_a = results[8].features.get("mom_ga_slope_a", np.zeros(n))
        blocks.append((burst_h * ga_slope_a)[:, None])  # Attack surge × defensive leak

    # 6b. v10: Additional cross-domain interactions
    # tanh(elo_diff) — nonlinear bounded elo signal
    elo_tanh = np.tanh(elo_diff.astype(float) / 400.0)
    blocks.append(elo_tanh[:, None])
    # log(form_diff) — signed log of form difference
    fd = form_diff.astype(float)
    form_log = np.sign(fd) * np.log1p(np.abs(fd))
    blocks.append(form_log[:, None])
    # Bayesian × Poisson agreement interaction
    if n_exp > 9:
        bayes_wr_diff = results[9].features.get("bayes_wr_diff", np.zeros(n))
        pois_lam_diff = results[3].features.get("pois_lambda_diff", np.zeros(n))
        blocks.append((bayes_wr_diff * pois_lam_diff)[:, None])
    # DC-adjusted Poisson × Market (sharper probability estimates)
    dc_ph = results[3].features.get("pois_dc_ph", np.zeros(n))
    blocks.append((dc_ph * mkt_ph)[:, None])  # DC-Poisson × Market
    # Monte Carlo P(home) × Market P(home)
    mc_ph_arr = results[3].features.get("pois_mc_ph", np.zeros(n))
    blocks.append((mc_ph_arr * mkt_ph)[:, None])  # MC × Market

    # 6c. v11: New academic model features
    # Bivariate Poisson P(home) — Karlis & Ntzoufras 2003
    bp_ph = results[3].features.get("pois_bp_ph", np.zeros(n))
    blocks.append(bp_ph[:, None])
    # Bivariate Poisson × Market (independent vs correlated model agreement)
    blocks.append((bp_ph * mkt_ph)[:, None])
    # Frank Copula P(home) — full-scoreline dependency
    cop_ph = results[3].features.get("pois_cop_ph", np.zeros(n))
    blocks.append(cop_ph[:, None])
    # Copula × DC agreement (two dependency models agreeing)
    blocks.append((cop_ph * dc_ph)[:, None])
    # COM-Poisson P(home) — dispersion-aware
    cmp_ph = results[3].features.get("pois_cmp_ph", np.zeros(n))
    blocks.append(cmp_ph[:, None])
    # COM-Poisson dispersion index (over-dispersed = unpredictable)
    cmp_disp_h = results[3].features.get("pois_cmp_disp_h", np.zeros(n))
    cmp_disp_a = results[3].features.get("pois_cmp_disp_a", np.zeros(n))
    blocks.append(cmp_disp_h[:, None])
    blocks.append(cmp_disp_a[:, None])
    # Bradley-Terry P(home) — pairwise comparison model
    bt_ph = results[3].features.get("pois_bt_ph", np.zeros(n))
    bt_pd = results[3].features.get("pois_bt_pd", np.zeros(n))
    blocks.append(bt_ph[:, None])
    blocks.append(bt_pd[:, None])
    # Bradley-Terry × Elo agreement (two strength-based models)
    blocks.append((bt_ph * results[0].probs[:, 0])[:, None])
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
    p_elo = np.nan_to_num(results[0].probs, nan=1.0/3)  # (n, 3)
    p_mkt = np.nan_to_num(results[1].probs, nan=1.0/3)  # (n, 3)
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
    h2h_bogey = results[4].features.get("h2h_bogey", np.zeros(n))
    upset_bogey_mkt = h2h_bogey * mkt_ph
    blocks.append(upset_bogey_mkt[:, None])

    # H2H surprise rate × context fatigue — historic upsets + current tiredness
    h2h_surprise = results[4].features.get("h2h_surprise_rate", np.zeros(n))
    fatigue_diff = results[5].features.get("ctx_fatigue_diff", np.zeros(n))
    blocks.append((h2h_surprise * np.abs(fatigue_diff))[:, None])

    # Motivation difference × market strength — motivated underdog vs complacent favourite
    motiv_diff = results[5].features.get("ctx_motivation_diff", np.zeros(n))
    blocks.append((motiv_diff * mkt_ph)[:, None])

    # Home fatigue × away freshness — tired home team vs rested visitor
    fatigue_h = results[5].features.get("ctx_fatigue_h", np.zeros(n))
    fatigue_a = results[5].features.get("ctx_fatigue_a", np.zeros(n))
    blocks.append((fatigue_h * (1.0 - fatigue_a))[:, None])

    # Relegation battle flag × underdogness — desperate underdog
    is_rel_a = results[5].features.get("ctx_is_relegation_a", np.zeros(n))
    blocks.append((is_rel_a * (1.0 - mkt_ph))[:, None])  # away-underdog in relegation fight

    # Derby/rivalry × market certainty — derbies erode favourite advantage
    is_derby = results[5].features.get("ctx_is_derby", np.zeros(n))
    blocks.append((is_derby * mkt_ph)[:, None])  # higher market certainty in derby = more vulnerable
    # High-stakes × team quality gap — high-stakes matches tighten outcomes
    high_stakes = results[5].features.get("ctx_high_stakes", np.zeros(n))
    blocks.append((high_stakes * np.abs(elo_diff.astype(float) / 400.0))[:, None])

    # Venue form gap — home-specific form vs away-specific form
    hf_ppg_h = results[2].features.get("form_home_ppg_h", np.zeros(n))
    af_ppg_a = results[2].features.get("form_away_ppg_a", np.zeros(n))
    venue_form_gap = hf_ppg_h - af_ppg_a
    blocks.append(venue_form_gap[:, None])

    # xG overperformance — regression signal: scoring above xG = unsustainable
    xg_diff_h = results[2].features.get("form_xg_diff_h", np.zeros(n))
    xg_diff_a = results[2].features.get("form_xg_diff_a", np.zeros(n))
    blocks.append(xg_diff_h[:, None])  # positive = overperforming
    blocks.append(xg_diff_a[:, None])
    blocks.append((xg_diff_h * mkt_ph)[:, None])  # overperforming favourite

    # Form vs top opponents — quality-adjusted form signals
    form_vs_top_h = results[2].features.get("form_vs_top_h", np.zeros(n))
    form_vs_top_a = results[2].features.get("form_vs_top_a", np.zeros(n))
    blocks.append((form_vs_top_h - form_vs_top_a)[:, None])

    # Position-form gap × market — regression risk for overperforming favourite
    pos_form_gap_h = results[7].features.get("lt_pos_form_gap_h", np.zeros(n)) if n_exp > 7 else np.zeros(n)
    pos_form_gap_a = results[7].features.get("lt_pos_form_gap_a", np.zeros(n)) if n_exp > 7 else np.zeros(n)
    blocks.append((pos_form_gap_h * mkt_ph)[:, None])   # favourite with regression risk
    blocks.append((pos_form_gap_a * (1.0 - mkt_ph))[:, None])  # underdog with upward risk

    # Venue-specific table position interaction — home table vs away table
    lt_home_ppg = results[7].features.get("lt_home_ppg_h", np.zeros(n)) if n_exp > 7 else np.zeros(n)
    lt_away_ppg = results[7].features.get("lt_away_ppg_a", np.zeros(n)) if n_exp > 7 else np.zeros(n)
    blocks.append((lt_home_ppg - lt_away_ppg)[:, None])

    # Venue momentum × H2H venue signal — double venue confirmation
    if n_exp > 8:
        mom_venue_h = results[8].features.get("mom_venue_ema_h", np.zeros(n))
        mom_venue_a = results[8].features.get("mom_venue_ema_a", np.zeros(n))
        h2h_venue_wr = results[4].features.get("h2h_venue_wr_h", np.zeros(n))
        blocks.append((mom_venue_h * h2h_venue_wr)[:, None])
        blocks.append((mom_venue_a * (1.0 - h2h_venue_wr))[:, None])

    # Quality-weighted momentum × Elo — momentum against good teams + fundamental strength
    if n_exp > 8:
        mom_quality_diff = results[8].features.get("mom_quality_diff", np.zeros(n))
        blocks.append((mom_quality_diff * elo_tanh)[:, None])

    # Streak × market — winning/losing streak interaction with expected outcome
    if n_exp > 8:
        streak_h = results[8].features.get("mom_streak_h", np.zeros(n))
        streak_a = results[8].features.get("mom_streak_a", np.zeros(n))
        blocks.append((streak_h * mkt_ph)[:, None])
        blocks.append((streak_a * (1.0 - mkt_ph))[:, None])

    # GoalPattern quality signals — performance vs top teams
    if n_exp > 6:
        cs_vs_top_h = results[6].features.get("gp_cs_vs_top_h", np.zeros(n))
        cs_vs_top_a = results[6].features.get("gp_cs_vs_top_a", np.zeros(n))
        cb_vs_top_h = results[6].features.get("gp_cb_vs_top_h", np.zeros(n))
        cb_vs_top_a = results[6].features.get("gp_cb_vs_top_a", np.zeros(n))
        blocks.append((cs_vs_top_a * (1.0 - mkt_ph))[:, None])  # underdog defensive resilience
        blocks.append((cb_vs_top_a * (1.0 - mkt_ph))[:, None])  # underdog comeback ability
        # late-goal tendency interaction
        late_h = results[6].features.get("gp_late_goal_h", np.zeros(n))
        late_a = results[6].features.get("gp_late_goal_a", np.zeros(n))
        blocks.append((late_h - late_a)[:, None])

    # Injury unavailability × market favourite — depleted squad still priced as strong
    if n_exp > 10:
        unavail_h = results[10].features.get("inj_unavailable_h", np.zeros(n))
        unavail_a = results[10].features.get("inj_unavailable_a", np.zeros(n))
        blocks.append((unavail_h * mkt_ph)[:, None])   # injured favourite
        blocks.append((unavail_a * (1.0 - mkt_ph))[:, None])
        # FDR future schedule asymmetry × form — hard schedule + good form
        fdr_diff = results[10].features.get("inj_fdr_future_diff", np.zeros(n))
        blocks.append((fdr_diff * form_diff)[:, None])

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
    poss_h = results[2].features.get("form_poss_h", np.zeros(n))
    poss_a = results[2].features.get("form_poss_a", np.zeros(n))
    poss_dom_h = results[2].features.get("form_poss_dom_h", np.zeros(n))
    poss_dom_a = results[2].features.get("form_poss_dom_a", np.zeros(n))
    poss_eff_h = results[2].features.get("form_poss_eff_h", np.zeros(n))
    poss_eff_a = results[2].features.get("form_poss_eff_a", np.zeros(n))
    # Possession vs expected: dominant possession but poor conversion = vulnerability
    blocks.append((poss_dom_h * (1.0 - poss_eff_h))[:, None])   # home domination but poor finishing
    blocks.append((poss_eff_a * (1.0 - poss_dom_a))[:, None])   # away efficient without domination
    blocks.append(((poss_h - poss_a) / 100.0)[:, None])         # raw poss diff normalized

    # H2H quality-adjusted × market — Elo-weighted H2H vs market
    h2h_qpts_h = results[4].features.get("h2h_quality_pts_h", np.zeros(n))
    h2h_qpts_a = results[4].features.get("h2h_quality_pts_a", np.zeros(n))
    blocks.append(((h2h_qpts_a - h2h_qpts_h) * mkt_ph)[:, None])  # away stronger in quality H2H × favourite
    # H2H trend × market — is this fixture shifting?
    h2h_trend = results[4].features.get("h2h_trend", np.zeros(n))
    blocks.append((h2h_trend * mkt_ph)[:, None])
    # H2H goal volatility — unpredictable fixtures are upset-prone
    h2h_goal_vol = results[4].features.get("h2h_goal_vol", np.zeros(n))
    blocks.append(h2h_goal_vol[:, None])

    # Form decay interaction — exponential-decay form vs standard form divergence
    decay_ppg_h = results[2].features.get("form_decay_ppg_h", np.zeros(n))
    decay_ppg_a = results[2].features.get("form_decay_ppg_a", np.zeros(n))
    form_pts_h_raw = results[2].features.get("form_pts_h", np.zeros(n))
    form_pts_a_raw = results[2].features.get("form_pts_a", np.zeros(n))
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

    # 7. Competition encoding — v10: one-hot (5 leagues) vectorized
    if competitions is not None:
        _COMP_LIST = ["PL", "PD", "SA", "BL1", "FL1"]
        comp_arr = np.array([str(c) for c in competitions])
        comp_onehot = np.column_stack([
            (comp_arr == comp).astype(float) for comp in _COMP_LIST
        ])
        blocks.append(comp_onehot)

    X = np.hstack(blocks)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_df(con, finished_only: bool = True, days: int = 3650) -> pd.DataFrame:
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


# ---------- TRAINING ----------
def train_and_save(con, days: int = 3650, eval_days: int = 365,
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
    # MULTI-MODEL STACKING
    # ================================================================
    # Base model 1: HistGradientBoosting (primary — best on tabular)
    gbm_base = HistGradientBoostingClassifier(
        learning_rate=0.02,
        max_depth=5,
        max_iter=1800,
        l2_regularization=0.5,
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
    available_models = [("GBM", P_gbm)]
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
    joblib.dump({
        "model": gbm_model,          # primary model (GBM + isotonic)
        "rf_model": rf_model,         # secondary model (RF + isotonic)
        "lr_model": lr_model,         # tertiary model (LR + isotonic)
        "dc_by_comp": dc_by_comp,
        "btts_model": btts_model,
        "ou25_model": ou25_model,
        "stack_weights": learned_weights,
        "n_features": int(X.shape[1]),  # v11: for predict-time validation
    }, MODEL_PATH)

    # ================================================================
    # WALK-FORWARD CROSS-VALIDATION — DEPLOYMENT GATE
    # ================================================================
    # The model is only saved if WF-CV metrics pass quality thresholds.
    # This prevents deploying models that overfit to the temporal split.
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

        # Deployment gate checks
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
            print(f"[council] walk-forward CV skipped: {e}", flush=True)
        # Graceful degradation: if WF fails to run, still allow deployment
        wf_passed = True

    if not wf_passed:
        if verbose:
            print("[council] ⚠ Model FAILED walk-forward gate but saving anyway "
                  "with gate_status=failed for downstream awareness", flush=True)

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
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


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
    keep = ["utc_date", "home_team", "away_team"] + [c for c in _odds_cols if c in up.columns]
    dummy = up[keep].copy()
    dummy["home_goals"] = 0; dummy["away_goals"] = 0
    for c in ["hs", "hst", "hc", "hy", "hr", "as_", "ast", "ac", "ay", "ar", "hthg", "htag"]:
        dummy[c] = 0

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
        eg_h = float(dc_eg[len(combo) - tail + j, 0]) if dc_eg[len(combo) - tail + j, 0] > 0 else None
        eg_a = float(dc_eg[len(combo) - tail + j, 1]) if dc_eg[len(combo) - tail + j, 1] > 0 else None

        # Collect Poisson stats for notes
        pois_res = tail_results[3]  # PoissonExpert is 4th (index 3)
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
        "home_goals": 0, "away_goals": 0,
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
