"""
Expert Council Model — Multi-expert ensemble for football match prediction.

Architecture:
    Layer 1 — Six specialist experts, each computing domain-specific features
              and analytical probability estimates:
              1. EloExpert        – Rating dynamics, team-specific home advantage
              2. MarketExpert     – Odds intelligence, line movement, sharp money
              3. FormExpert       – Opposition-adjusted rolling form
              4. PoissonExpert    – Attack / defence + venue-split + score matrix
              5. H2HExpert        – Bayesian head-to-head with time decay
              6. ContextExpert    – Rest / congestion / calendar / season stage
    Layer 2 — Conflict & consensus signals derived from expert disagreement
    Layer 3 — Meta-learner (HistGradientBoosting + isotonic calibration)
    Layer 4 — Ollama interpreter (optional) for narrative match analysis

v7 Innovations:
    • PoissonExpert v2: venue-specific attack/defense, BTTS/O2.5/O1.5/U2.5
      probabilities, most likely score, goal-diff skewness, zero-inflation
    • ContextExpert v2: season progress, early/late season flags, day-of-week,
      weekend/midweek detection, hour of kickoff, 30-day congestion,
      rest ratio, short-rest flags
    • Meta-learner v2: 5 pairwise expert agreement features, max disagreement,
      winner vote concentration, isotonic calibration (cv=5)
    • Tuned hyperparameters: lr=0.02, depth=5, iter=1800, L2=0.5, leaf=50

Prior innovations (retained):
    • Opposition-Adjusted Form (OAF)  — weights results by opponent Elo
    • Bayesian H2H with Dirichlet prior — handles sparse matchups
    • Expert confidence calibration — dynamic weighting by data availability
    • Conflict signals as features — expert disagreement informs uncertainty
    • Elo momentum + stability features
    • Fixture congestion index
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from footy.config import settings
from footy.normalize import canonical_team_name
from footy.models.dixon_coles import fit_dc, predict_1x2, DCModel

MODEL_VERSION = "v7_council"
MODEL_PATH = Path("data/models") / f"{MODEL_VERSION}.joblib"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _f(x) -> float:
    """Safe float cast."""
    try:
        if x is None:
            return 0.0
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _raw(raw_json) -> dict:
    if raw_json is None:
        return {}
    if isinstance(raw_json, dict):
        return raw_json
    try:
        return json.loads(raw_json)
    except Exception:
        return {}


def _pts(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _label(hg: int, ag: int) -> int:
    if hg > ag:
        return 0
    if hg == ag:
        return 1
    return 2


def _entropy3(p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _norm3(a, b, c) -> tuple[float, float, float]:
    s = a + b + c
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def _implied(h, d, a):
    h, d, a = _f(h), _f(d), _f(a)
    if h <= 1 or d <= 1 or a <= 1:
        return (0.0, 0.0, 0.0, 0.0)
    ih, id_, ia = 1 / h, 1 / d, 1 / a
    s = ih + id_ + ia
    return (ih / s, id_ / s, ia / s, s - 1.0)


# ---------------------------------------------------------------------------
# Expert result container
# ---------------------------------------------------------------------------
@dataclass
class ExpertResult:
    """Output from a single expert for n matches."""
    probs: np.ndarray          # (n, 3) analytical P(H, D, A)
    confidence: np.ndarray     # (n,) ∈ [0, 1]
    features: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract expert
# ---------------------------------------------------------------------------
class Expert(ABC):
    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> ExpertResult:
        """Compute expert features over matches sorted by utc_date ASC.

        ``df`` has columns: utc_date, home_team, away_team, home_goals,
        away_goals (for finished) or 0 placeholders (for upcoming dummies).
        Plus optional extras columns (hs, hst, ...).
        """
        ...


# ===================================================================
# 1. ELO EXPERT
# ===================================================================
class EloExpert(Expert):
    """
    Team Elo ratings with:
    - Team-specific home advantage (tracked from home/away records)
    - Dynamic K-factor (convergence + goal-diff scaling)
    - Elo momentum (direction of recent Elo change)
    - Elo volatility (std-dev of rating over last N games)
    """
    name = "elo"

    ELO_DEFAULT = 1500.0
    K_BASE = 20.0
    BLANKET_HOME = 40.0
    TRACK_N = 20          # home / away record window
    MOMENTUM_N = 6        # games for momentum calc

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        ratings: dict[str, float] = {}
        counts: dict[str, int] = {}
        home_rec: dict[str, list] = {}    # (pts, gf, ga)
        away_rec: dict[str, list] = {}
        elo_history: dict[str, list] = {} # last N ratings for volatility

        out_elo_h = np.zeros(n)
        out_elo_a = np.zeros(n)
        out_diff = np.zeros(n)
        out_ph = np.zeros(n)
        out_pd = np.zeros(n)
        out_pa = np.zeros(n)
        out_conf = np.zeros(n)
        out_home_adv = np.zeros(n)
        out_momentum_h = np.zeros(n)
        out_momentum_a = np.zeros(n)
        out_volatility_h = np.zeros(n)
        out_volatility_a = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            rh = ratings.get(h, self.ELO_DEFAULT)
            ra = ratings.get(a, self.ELO_DEFAULT)

            # team-specific home advantage
            h_rec = home_rec.get(h, [])
            a_rec_h = away_rec.get(h, [])
            if len(h_rec) >= 3 and len(a_rec_h) >= 3:
                h_ppg = np.mean([x[0] for x in h_rec[-self.TRACK_N:]])
                a_ppg = np.mean([x[0] for x in a_rec_h[-self.TRACK_N:]])
                h_gd = np.mean([x[1] - x[2] for x in h_rec[-self.TRACK_N:]])
                a_gd = np.mean([x[1] - x[2] for x in a_rec_h[-self.TRACK_N:]])
                adv = self.BLANKET_HOME + (h_ppg - a_ppg) * 50 + (h_gd - a_gd) * 10
                adv = max(0.0, min(150.0, adv))
            else:
                adv = self.BLANKET_HOME

            rh_adj = rh + adv

            # expected
            e_h = 1.0 / (1.0 + 10.0 ** (-(rh_adj - ra) / 400.0))
            diff = abs(rh_adj - ra)
            p_draw = max(0.18, min(0.34, 0.26 + 0.06 * math.exp(-diff / 200)))
            ph = e_h * (1 - p_draw)
            pa = (1 - e_h) * (1 - p_draw)
            ph, pd_, pa = _norm3(ph, p_draw, pa)

            # momentum & volatility
            hist_h = elo_history.get(h, [])
            hist_a = elo_history.get(a, [])
            mom_h = (hist_h[-1] - hist_h[-self.MOMENTUM_N]) / max(1, len(hist_h[-self.MOMENTUM_N:])) if len(hist_h) >= self.MOMENTUM_N else 0.0
            mom_a = (hist_a[-1] - hist_a[-self.MOMENTUM_N]) / max(1, len(hist_a[-self.MOMENTUM_N:])) if len(hist_a) >= self.MOMENTUM_N else 0.0
            vol_h = float(np.std(hist_h[-self.MOMENTUM_N:])) if len(hist_h) >= 3 else 30.0
            vol_a = float(np.std(hist_a[-self.MOMENTUM_N:])) if len(hist_a) >= 3 else 30.0

            # confidence — increases with games played (capped at 1)
            c_h = min(1.0, counts.get(h, 0) / 30.0)
            c_a = min(1.0, counts.get(a, 0) / 30.0)
            conf = (c_h + c_a) / 2.0

            out_elo_h[i] = rh; out_elo_a[i] = ra; out_diff[i] = rh - ra
            out_ph[i] = ph; out_pd[i] = pd_; out_pa[i] = pa
            out_conf[i] = conf; out_home_adv[i] = adv
            out_momentum_h[i] = mom_h; out_momentum_a[i] = mom_a
            out_volatility_h[i] = vol_h; out_volatility_a[i] = vol_a

            # --- update state AFTER recording pre-match values ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            gd = abs(hg - ag)
            n_h = counts.get(h, 0)
            n_a = counts.get(a, 0)
            conv_h = 1.0 + max(0.0, 1.0 - n_h / 60.0)
            conv_a = 1.0 + max(0.0, 1.0 - n_a / 60.0)
            gd_m = 1.0 if gd <= 1 else (1.15 if gd == 2 else min(1.5, 1.3 + 0.05 * (gd - 3)))
            k = self.K_BASE * ((conv_h + conv_a) / 2) * gd_m
            delta = k * (s_home - e_h)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
            counts[h] = n_h + 1
            counts[a] = n_a + 1
            home_rec.setdefault(h, []).append((_pts(hg, ag), hg, ag))
            away_rec.setdefault(a, []).append((_pts(ag, hg), ag, hg))
            elo_history.setdefault(h, []).append(ratings[h])
            elo_history.setdefault(a, []).append(ratings[a])

        return ExpertResult(
            probs=np.column_stack([out_ph, out_pd, out_pa]),
            confidence=out_conf,
            features={
                "elo_home": out_elo_h, "elo_away": out_elo_a, "elo_diff": out_diff,
                "elo_home_adv": out_home_adv,
                "elo_momentum_h": out_momentum_h, "elo_momentum_a": out_momentum_a,
                "elo_volatility_h": out_volatility_h, "elo_volatility_a": out_volatility_a,
            },
        )


# ===================================================================
# 2. MARKET EXPERT
# ===================================================================
class MarketExpert(Expert):
    """
    Odds intelligence: implied probabilities from multiple bookmaker tiers,
    opening→closing line movement (sharp money signal), overround as
    uncertainty proxy.
    """
    name = "market"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        ph = np.zeros(n); pd_ = np.zeros(n); pa = np.zeros(n)
        overround = np.zeros(n)
        src_quality = np.zeros(n)  # 3=closing, 2=avg, 1=max, 0=primary
        has_odds = np.zeros(n)
        move_h = np.zeros(n); move_d = np.zeros(n); move_a = np.zeros(n)
        has_move = np.zeros(n)
        ou25 = np.zeros(n); ou_over = np.zeros(n); has_ou = np.zeros(n)
        conf = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            raw = _raw(getattr(r, "raw_json", None))
            b365h = _f(getattr(r, "b365h", None))
            b365d = _f(getattr(r, "b365d", None))
            b365a = _f(getattr(r, "b365a", None))

            # Read structured columns first, fall back to raw_json
            b365ch = _f(getattr(r, "b365ch", None)) or _f(raw.get("B365CH"))
            b365cd = _f(getattr(r, "b365cd", None)) or _f(raw.get("B365CD"))
            b365ca = _f(getattr(r, "b365ca", None)) or _f(raw.get("B365CA"))
            avgh_v = _f(getattr(r, "avgh", None)) or _f(raw.get("AvgH"))
            avgd_v = _f(getattr(r, "avgd", None)) or _f(raw.get("AvgD"))
            avga_v = _f(getattr(r, "avga", None)) or _f(raw.get("AvgA"))
            maxh_v = _f(getattr(r, "maxh", None)) or _f(raw.get("MaxH"))
            maxd_v = _f(getattr(r, "maxd", None)) or _f(raw.get("MaxD"))
            maxa_v = _f(getattr(r, "maxa", None)) or _f(raw.get("MaxA"))
            psh_v = _f(getattr(r, "psh", None)) or _f(raw.get("PSH"))
            psd_v = _f(getattr(r, "psd", None)) or _f(raw.get("PSD"))
            psa_v = _f(getattr(r, "psa", None)) or _f(raw.get("PSA"))

            # best available odds (closing > Pinnacle > avg > max > primary)
            for trio, sq in [
                ((b365ch, b365cd, b365ca), 4.0),
                ((psh_v, psd_v, psa_v), 3.0),
                ((avgh_v, avgd_v, avga_v), 2.0),
                ((maxh_v, maxd_v, maxa_v), 1.0),
                ((b365h, b365d, b365a), 0.0),
            ]:
                imp = _implied(trio[0], trio[1], trio[2])
                if imp[0] > 0:
                    ph[i], pd_[i], pa[i] = imp[0], imp[1], imp[2]
                    overround[i] = imp[3]
                    src_quality[i] = sq
                    has_odds[i] = 1.0
                    break

            # line movement (opening → closing)
            imp_o = _implied(b365h, b365d, b365a)
            imp_c = _implied(b365ch, b365cd, b365ca)
            if imp_o[0] > 0 and imp_c[0] > 0:
                move_h[i] = imp_c[0] - imp_o[0]
                move_d[i] = imp_c[1] - imp_o[1]
                move_a[i] = imp_c[2] - imp_o[2]
                has_move[i] = 1.0

            # over/under 2.5 — read from proper columns first
            avg_o25_v = _f(getattr(r, "avg_o25", None)) or _f(raw.get("AvgC>2.5")) or _f(raw.get("Avg>2.5"))
            avg_u25_v = _f(getattr(r, "avg_u25", None)) or _f(raw.get("AvgC<2.5")) or _f(raw.get("Avg<2.5"))
            b365_o25_v = _f(getattr(r, "b365_o25", None)) or _f(raw.get("B365C>2.5")) or _f(raw.get("B365>2.5"))
            b365_u25_v = _f(getattr(r, "b365_u25", None)) or _f(raw.get("B365C<2.5")) or _f(raw.get("B365<2.5"))
            max_o25_v = _f(getattr(r, "max_o25", None)) or _f(raw.get("Max>2.5"))
            max_u25_v = _f(getattr(r, "max_u25", None)) or _f(raw.get("Max<2.5"))

            for o_, u_ in [
                (avg_o25_v, avg_u25_v),
                (b365_o25_v, b365_u25_v),
                (max_o25_v, max_u25_v),
            ]:
                if o_ and u_ and o_ > 1 and u_ > 1:
                    io, iu = 1 / o_, 1 / u_
                    ou25[i] = io / (io + iu)
                    ou_over[i] = io + iu - 1
                    has_ou[i] = 1.0
                    break

            # confidence based on data availability
            c = 0.3 * has_odds[i] + 0.3 * min(1.0, src_quality[i] / 3.0) + 0.2 * has_move[i] + 0.2 * has_ou[i]
            conf[i] = c

        return ExpertResult(
            probs=np.column_stack([ph, pd_, pa]),
            confidence=conf,
            features={
                "mkt_overround": overround, "mkt_src_quality": src_quality,
                "mkt_has_odds": has_odds,
                "mkt_move_h": move_h, "mkt_move_d": move_d, "mkt_move_a": move_a,
                "mkt_has_move": has_move,
                "mkt_ou25": ou25, "mkt_ou_over": ou_over, "mkt_has_ou": has_ou,
            },
        )


# ===================================================================
# 3. FORM EXPERT
# ===================================================================
class FormExpert(Expert):
    """
    Rolling form analysis with Opposition-Adjusted Form (OAF):
    - Standard rolling averages (goals, points, shots, corners, cards)
    - OAF: weight each result by opponent Elo percentile
    - Venue-split form (home-only PPG, away-only PPG)
    - Momentum (acceleration), streak, clean-sheet / BTTS rates
    """
    name = "form"

    ROLL = 5              # standard rolling window
    OAF_WINDOW = 10       # opponent-adjusted form window
    EXTENDED_WINDOW = 10  # for rates (CS, BTTS)

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        roll: dict[str, list[dict]] = {}
        elo: dict[str, float] = {}  # track Elo for OAF weighting
        home_ppg: dict[str, list[float]] = {}
        away_ppg: dict[str, list[float]] = {}
        ELO_MED = 1500.0

        # outputs
        gf_h = np.zeros(n); ga_h = np.zeros(n); pp_h = np.zeros(n)
        gf_a = np.zeros(n); ga_a = np.zeros(n); pp_a = np.zeros(n)
        sh_h = np.zeros(n); sot_h = np.zeros(n); cor_h = np.zeros(n); card_h = np.zeros(n)
        sh_a = np.zeros(n); sot_a = np.zeros(n); cor_a = np.zeros(n); card_a = np.zeros(n)
        oaf_h = np.zeros(n); oaf_a = np.zeros(n)    # opposition-adjusted PPG
        hf_ppg_h = np.zeros(n); af_ppg_a = np.zeros(n)  # home-form PPG, away-form PPG
        cs_h = np.zeros(n); cs_a = np.zeros(n)
        btts_h = np.zeros(n); btts_a = np.zeros(n)
        mom_h = np.zeros(n); mom_a = np.zeros(n)
        strk_h = np.zeros(n); strk_a = np.zeros(n)
        gsup_h = np.zeros(n); gsup_a = np.zeros(n)
        sotr_h = np.zeros(n); sotr_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)

        def _avg(team, key, w=None):
            xs = [d.get(key, 0.0) for d in roll.get(team, [])[-( w or self.ROLL):]]
            return float(np.mean(xs)) if xs else 0.0

        def _rate(team, key):
            xs = [d.get(key, 0.0) for d in roll.get(team, [])[-self.EXTENDED_WINDOW:]]
            return float(np.mean(xs)) if xs else 0.0

        def _oaf_ppg(team):
            """Opposition-adjusted PPG: weight points by opponent Elo."""
            recs = roll.get(team, [])[-self.OAF_WINDOW:]
            if not recs:
                return 0.0
            total_w = 0.0; total_p = 0.0
            for d in recs:
                opp_elo = d.get("opp_elo", ELO_MED)
                w = opp_elo / ELO_MED  # >1 for strong opponents, <1 for weak
                total_p += d["pts"] * w
                total_w += w
            return total_p / max(total_w, 1e-6)

        def _momentum(team):
            hist = roll.get(team, [])
            if len(hist) < 4:
                return 0.0
            recent = np.mean([d["pts"] for d in hist[-3:]])
            older = np.mean([d["pts"] for d in hist[-6:]])
            return float(recent - older)

        def _streak(team):
            hist = roll.get(team, [])
            if not hist:
                return 0.0
            s = 0
            last = hist[-1]["pts"]
            for d in reversed(hist):
                if last == 3 and d["pts"] == 3:
                    s += 1
                elif last == 0 and d["pts"] == 0:
                    s -= 1
                else:
                    break
            return float(s)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            # pre-match features
            gf_h[i] = _avg(h, "gf"); ga_h[i] = _avg(h, "ga"); pp_h[i] = _avg(h, "pts")
            gf_a[i] = _avg(a, "gf"); ga_a[i] = _avg(a, "ga"); pp_a[i] = _avg(a, "pts")
            sh_h[i] = _avg(h, "sh"); sot_h[i] = _avg(h, "sot"); cor_h[i] = _avg(h, "cor"); card_h[i] = _avg(h, "card")
            sh_a[i] = _avg(a, "sh"); sot_a[i] = _avg(a, "sot"); cor_a[i] = _avg(a, "cor"); card_a[i] = _avg(a, "card")
            oaf_h[i] = _oaf_ppg(h); oaf_a[i] = _oaf_ppg(a)

            # venue-split
            hrecs = home_ppg.get(h, [])
            arecs = away_ppg.get(a, [])
            hf_ppg_h[i] = float(np.mean(hrecs[-self.ROLL:])) if hrecs else 0.0
            af_ppg_a[i] = float(np.mean(arecs[-self.ROLL:])) if arecs else 0.0

            cs_h[i] = _rate(h, "cs"); cs_a[i] = _rate(a, "cs")
            btts_h[i] = _rate(h, "btts"); btts_a[i] = _rate(a, "btts")
            mom_h[i] = _momentum(h); mom_a[i] = _momentum(a)
            strk_h[i] = _streak(h); strk_a[i] = _streak(a)
            gsup_h[i] = _avg(h, "gsup"); gsup_a[i] = _avg(a, "gsup")
            sv_h = _avg(h, "sh"); sv_sot_h = _avg(h, "sot")
            sv_a = _avg(a, "sh"); sv_sot_a = _avg(a, "sot")
            sotr_h[i] = sv_sot_h / max(sv_h, 0.1)
            sotr_a[i] = sv_sot_a / max(sv_a, 0.1)

            # form-based probs (OAF comparison)
            oaf_diff = oaf_h[i] - oaf_a[i]
            p_h = 1 / (1 + math.exp(-0.8 * oaf_diff))
            p_a = 1 - p_h
            p_d = max(0.18, 0.28 - abs(oaf_diff) * 0.05)
            probs[i] = _norm3(p_h * (1 - p_d), p_d, p_a * (1 - p_d))

            # confidence
            n_h = len(roll.get(h, []))
            n_a = len(roll.get(a, []))
            conf[i] = min(1.0, (n_h + n_a) / 20.0)

            # --- update state ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            hs = _f(getattr(r, "hs", 0)); hst = _f(getattr(r, "hst", 0))
            hc = _f(getattr(r, "hc", 0)); hy = _f(getattr(r, "hy", 0))
            hr_ = _f(getattr(r, "hr", 0))
            as_ = _f(getattr(r, "as_", 0)); ast = _f(getattr(r, "ast", 0))
            ac = _f(getattr(r, "ac", 0)); ay = _f(getattr(r, "ay", 0))
            ar_ = _f(getattr(r, "ar", 0))

            opp_elo_a = elo.get(a, 1500.0)
            opp_elo_h = elo.get(h, 1500.0)

            roll.setdefault(h, []).append({
                "gf": hg, "ga": ag, "pts": _pts(hg, ag),
                "sh": hs, "sot": hst, "cor": hc, "card": hy + 2.5 * hr_,
                "cs": 1.0 if ag == 0 else 0.0,
                "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
                "gsup": hg - ag, "opp_elo": opp_elo_a,
            })
            roll.setdefault(a, []).append({
                "gf": ag, "ga": hg, "pts": _pts(ag, hg),
                "sh": as_, "sot": ast, "cor": ac, "card": ay + 2.5 * ar_,
                "cs": 1.0 if hg == 0 else 0.0,
                "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
                "gsup": ag - hg, "opp_elo": opp_elo_h,
            })
            home_ppg.setdefault(h, []).append(float(_pts(hg, ag)))
            away_ppg.setdefault(a, []).append(float(_pts(ag, hg)))
            # lightweight Elo update for OAF weighting
            rh = elo.get(h, 1500.0); ra = elo.get(a, 1500.0)
            e = 1 / (1 + 10 ** (-(rh + 40 - ra) / 400))
            s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = 16 * (s - e)
            elo[h] = rh + delta; elo[a] = ra - delta

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "form_gf_h": gf_h, "form_ga_h": ga_h, "form_pts_h": pp_h,
                "form_gf_a": gf_a, "form_ga_a": ga_a, "form_pts_a": pp_a,
                "form_sh_h": sh_h, "form_sot_h": sot_h, "form_cor_h": cor_h, "form_card_h": card_h,
                "form_sh_a": sh_a, "form_sot_a": sot_a, "form_cor_a": cor_a, "form_card_a": card_a,
                "form_oaf_h": oaf_h, "form_oaf_a": oaf_a,
                "form_home_ppg_h": hf_ppg_h, "form_away_ppg_a": af_ppg_a,
                "form_cs_h": cs_h, "form_cs_a": cs_a,
                "form_btts_h": btts_h, "form_btts_a": btts_a,
                "form_momentum_h": mom_h, "form_momentum_a": mom_a,
                "form_streak_h": strk_h, "form_streak_a": strk_a,
                "form_gsup_h": gsup_h, "form_gsup_a": gsup_a,
                "form_sotr_h": sotr_h, "form_sotr_a": sotr_a,
            },
        )


# ===================================================================
# 4. POISSON EXPERT
# ===================================================================
class PoissonExpert(Expert):
    """
    Attack / defence strength via EMA tracking → Poisson goal distribution →
    analytical P(H, D, A) through score matrix convolution.
    """
    name = "poisson"

    ALPHA = 0.08        # EMA smoothing factor
    MAX_GOALS = 8       # max goals to consider in score matrix
    AVG = 1.35          # league average goals per team
    VENUE_ALPHA = 0.06  # slower EMA for venue-specific stats

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        # venue-specific attack/defense
        home_att: dict[str, float] = {}
        home_def: dict[str, float] = {}
        away_att: dict[str, float] = {}
        away_def: dict[str, float] = {}
        game_count: dict[str, int] = {}

        lam_h = np.zeros(n); lam_a = np.zeros(n)
        probs = np.full((n, 3), 1 / 3)
        conf = np.zeros(n)
        # new score-derived features
        pois_btts = np.zeros(n)        # P(both teams to score)
        pois_o25 = np.zeros(n)         # P(over 2.5 goals)
        pois_o15 = np.zeros(n)         # P(over 1.5 goals)
        pois_u25 = np.zeros(n)         # P(under 2.5 goals)
        pois_cs00 = np.zeros(n)        # P(0-0 clean sheet)
        pois_ml_hg = np.zeros(n)       # most likely home goals
        pois_ml_ag = np.zeros(n)       # most likely away goals
        pois_zero_h = np.zeros(n)      # P(home scores 0)
        pois_zero_a = np.zeros(n)      # P(away scores 0)
        pois_skew = np.zeros(n)        # skewness of goal diff distribution

        def _a(t): return attack.get(t, self.AVG)
        def _d(t): return defense.get(t, self.AVG * 0.9)

        from scipy.stats import poisson as poisson_dist

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            ah, dh = _a(h), _d(h)
            aa, da = _a(a), _d(a)

            # blend overall + venue-specific if available
            ha_h = home_att.get(h, ah)
            hd_h = home_def.get(h, dh)
            aa_a = away_att.get(a, aa)
            ad_a = away_def.get(a, da)
            att_h = 0.7 * ah + 0.3 * ha_h  # weighted blend
            def_h = 0.7 * dh + 0.3 * hd_h
            att_a = 0.7 * aa + 0.3 * aa_a
            def_a = 0.7 * da + 0.3 * ad_a

            l_h = max(0.3, min(4.5, att_h * def_a / self.AVG * 1.05))
            l_a = max(0.3, min(4.5, att_a * def_h / self.AVG))
            lam_h[i] = l_h; lam_a[i] = l_a

            # score matrix
            goals_range = np.arange(self.MAX_GOALS + 1)
            ph_dist = poisson_dist.pmf(goals_range, l_h)
            pa_dist = poisson_dist.pmf(goals_range, l_a)
            score_mx = np.outer(ph_dist, pa_dist)
            p_home = float(np.sum(np.tril(score_mx, -1)))
            p_draw = float(np.sum(np.diag(score_mx)))
            p_away = float(np.sum(np.triu(score_mx, 1)))
            ph_, pd__, pa_ = _norm3(p_home, p_draw, p_away)
            probs[i] = [ph_, pd__, pa_]

            # score-derived features from the score matrix
            pois_cs00[i] = float(score_mx[0, 0])
            pois_zero_h[i] = float(ph_dist[0])
            pois_zero_a[i] = float(pa_dist[0])
            pois_btts[i] = float(1.0 - ph_dist[0] - pa_dist[0] + score_mx[0, 0])
            # over/under
            total_under25 = sum(
                float(score_mx[hg, ag])
                for hg in range(self.MAX_GOALS + 1)
                for ag in range(self.MAX_GOALS + 1)
                if hg + ag <= 2
            )
            total_under15 = sum(
                float(score_mx[hg, ag])
                for hg in range(self.MAX_GOALS + 1)
                for ag in range(self.MAX_GOALS + 1)
                if hg + ag <= 1
            )
            pois_u25[i] = total_under25
            pois_o25[i] = 1.0 - total_under25
            pois_o15[i] = 1.0 - total_under15
            # most likely score
            best_idx = np.unravel_index(np.argmax(score_mx), score_mx.shape)
            pois_ml_hg[i] = best_idx[0]; pois_ml_ag[i] = best_idx[1]
            # goal difference skewness (positive = home-heavy)
            gd_probs = np.zeros(2 * self.MAX_GOALS + 1)
            for hg_i in range(self.MAX_GOALS + 1):
                for ag_i in range(self.MAX_GOALS + 1):
                    gd_probs[hg_i - ag_i + self.MAX_GOALS] += score_mx[hg_i, ag_i]
            gd_range = np.arange(-self.MAX_GOALS, self.MAX_GOALS + 1)
            mu = np.sum(gd_range * gd_probs)
            var = np.sum(gd_probs * (gd_range - mu) ** 2)
            std = max(np.sqrt(var), 1e-6)
            pois_skew[i] = float(np.sum(gd_probs * ((gd_range - mu) / std) ** 3))

            # confidence — increases with number of known teams
            gc_h = game_count.get(h, 0)
            gc_a = game_count.get(a, 0)
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # --- update ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            if h in attack:
                attack[h] = (1 - self.ALPHA) * ah + self.ALPHA * hg
                defense[h] = (1 - self.ALPHA) * dh + self.ALPHA * ag
            else:
                attack[h] = float(hg) if hg > 0 else self.AVG
                defense[h] = float(ag) if ag > 0 else self.AVG * 0.9
            if a in attack:
                attack[a] = (1 - self.ALPHA) * aa + self.ALPHA * ag
                defense[a] = (1 - self.ALPHA) * da + self.ALPHA * hg
            else:
                attack[a] = float(ag) if ag > 0 else self.AVG
                defense[a] = float(hg) if hg > 0 else self.AVG * 0.9
            # venue-specific EMA
            if h in home_att:
                home_att[h] = (1 - self.VENUE_ALPHA) * home_att.get(h, ah) + self.VENUE_ALPHA * hg
                home_def[h] = (1 - self.VENUE_ALPHA) * home_def.get(h, dh) + self.VENUE_ALPHA * ag
            else:
                home_att[h] = float(hg) if hg > 0 else self.AVG
                home_def[h] = float(ag) if ag > 0 else self.AVG * 0.9
            if a in away_att:
                away_att[a] = (1 - self.VENUE_ALPHA) * away_att.get(a, aa) + self.VENUE_ALPHA * ag
                away_def[a] = (1 - self.VENUE_ALPHA) * away_def.get(a, da) + self.VENUE_ALPHA * hg
            else:
                away_att[a] = float(ag) if ag > 0 else self.AVG
                away_def[a] = float(hg) if hg > 0 else self.AVG * 0.9
            game_count[h] = game_count.get(h, 0) + 1
            game_count[a] = game_count.get(a, 0) + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "pois_lambda_h": lam_h, "pois_lambda_a": lam_a,
                "pois_lambda_diff": lam_h - lam_a,
                "pois_lambda_sum": lam_h + lam_a,
                "pois_lambda_prod": lam_h * lam_a,
                "pois_btts": pois_btts, "pois_o25": pois_o25,
                "pois_o15": pois_o15, "pois_u25": pois_u25,
                "pois_cs00": pois_cs00,
                "pois_zero_h": pois_zero_h, "pois_zero_a": pois_zero_a,
                "pois_ml_hg": pois_ml_hg, "pois_ml_ag": pois_ml_ag,
                "pois_skew": pois_skew,
            },
        )


# ===================================================================
# 5. H2H EXPERT
# ===================================================================
class H2HExpert(Expert):
    """
    Bayesian head-to-head:
    - Dirichlet prior calibrated to league base rates
    - Time-decayed observations (half-life 730 days)
    - Venue-specific sub-analysis
    - Proper uncertainty quantification
    """
    name = "h2h"

    HALF_LIFE = 730.0       # days
    PRIOR_STRENGTH = 3.0    # pseudo-count weight of prior
    # base-rate priors (home-centric)
    PRIOR_H = 0.45
    PRIOR_D = 0.27
    PRIOR_A = 0.28

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        decay = math.log(2) / self.HALF_LIFE
        h2h: dict[frozenset, list] = {}         # (home, away, hg, ag, dt)
        venue: dict[tuple, list] = {}            # (home, away) -> [(hg, ag, dt)]

        probs = np.full((n, 3), [self.PRIOR_H, self.PRIOR_D, self.PRIOR_A])
        conf = np.zeros(n)
        out_n = np.zeros(n)
        out_pts_h = np.zeros(n); out_pts_a = np.zeros(n); out_gd = np.zeros(n)
        out_vn = np.zeros(n); out_vpts = np.zeros(n); out_vgd = np.zeros(n)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            dt = r.utc_date
            key = frozenset([h, a])
            past = h2h.get(key, [])

            # Bayesian update with time-decayed counts
            alpha_h = self.PRIOR_STRENGTH * self.PRIOR_H
            alpha_d = self.PRIOR_STRENGTH * self.PRIOR_D
            alpha_a = self.PRIOR_STRENGTH * self.PRIOR_A
            total_w = 0.0
            pts_home_w = pts_away_w = gd_w = 0.0

            for phome, paway, hg_p, ag_p, pdt in past[-24:]:
                age = max(0.0, (dt - pdt).total_seconds() / 86400.0)
                w = math.exp(-decay * age)
                # determine outcome from perspective of home team in *this* match
                if phome == h and paway == a:
                    gh, ga = hg_p, ag_p
                else:
                    gh, ga = ag_p, hg_p

                if gh > ga:
                    alpha_h += w
                elif gh == ga:
                    alpha_d += w
                else:
                    alpha_a += w

                pts_home_w += _pts(gh, ga) * w
                pts_away_w += _pts(ga, gh) * w
                gd_w += (gh - ga) * w
                total_w += w

            s = alpha_h + alpha_d + alpha_a
            probs[i] = [alpha_h / s, alpha_d / s, alpha_a / s]

            n_raw = len(past[-24:])
            out_n[i] = n_raw
            out_pts_h[i] = pts_home_w / max(total_w, 1e-6)
            out_pts_a[i] = pts_away_w / max(total_w, 1e-6)
            out_gd[i] = gd_w / max(total_w, 1e-6)

            # venue-specific
            vpast = venue.get((h, a), [])[-6:]
            vn = len(vpast)
            vpts = vgd = 0.0
            for hg_v, ag_v, _ in vpast:
                vpts += _pts(hg_v, ag_v)
                vgd += (hg_v - ag_v)
            out_vn[i] = vn
            out_vpts[i] = vpts / max(1, vn)
            out_vgd[i] = vgd / max(1, vn)

            # confidence = f(effective sample size)
            eff_n = total_w  # time-weighted effective sample size
            conf[i] = min(1.0, eff_n / 8.0)

            # --- update ---
            hg, ag = int(r.home_goals), int(r.away_goals)
            h2h.setdefault(key, []).append((h, a, hg, ag, dt))
            venue.setdefault((h, a), []).append((hg, ag, dt))

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "h2h_n": out_n, "h2h_pts_h": out_pts_h, "h2h_pts_a": out_pts_a,
                "h2h_gd": out_gd,
                "h2h_venue_n": out_vn, "h2h_venue_pts": out_vpts, "h2h_venue_gd": out_vgd,
            },
        )


# ===================================================================
# 6. CONTEXT EXPERT
# ===================================================================
class ContextExpert(Expert):
    """
    Rest days, fixture congestion, season calendar, and derived contextual
    features.
    """
    name = "context"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        last_date: dict[str, Any] = {}
        match_dates: dict[str, list] = {}   # for congestion calc

        rest_h = np.zeros(n); rest_a = np.zeros(n)
        cong7_h = np.zeros(n); cong7_a = np.zeros(n)
        cong14_h = np.zeros(n); cong14_a = np.zeros(n)
        cong30_h = np.zeros(n); cong30_a = np.zeros(n)
        # season calendar features
        season_prog = np.zeros(n)       # 0-1 season progress (Aug=0, May=1)
        is_early = np.zeros(n)          # first 6 GWs proxy
        is_late = np.zeros(n)           # final 6 GWs proxy
        dow = np.zeros(n)               # day of week (0=Mon, 6=Sun)
        is_weekend = np.zeros(n)        # Sat/Sun
        is_midweek = np.zeros(n)        # Tue/Wed (European nights)
        hour_utc = np.zeros(n)          # hour of kickoff
        # fatigue interactions
        rest_ratio = np.zeros(n)        # home/away rest ratio
        short_rest_h = np.zeros(n)      # home on <3 days rest
        short_rest_a = np.zeros(n)      # away on <3 days rest

        probs = np.full((n, 3), [0.44, 0.28, 0.28])  # mild home prior
        conf = np.full(n, 0.3)  # context is informative but never decisive

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            dt = r.utc_date

            # rest days
            lh = last_date.get(h)
            la = last_date.get(a)
            rh = 7.0 if lh is None else max(0.0, (dt - lh).total_seconds() / 86400.0)
            ra = 7.0 if la is None else max(0.0, (dt - la).total_seconds() / 86400.0)
            rest_h[i] = rh; rest_a[i] = ra
            rest_ratio[i] = rh / max(ra, 0.5)
            short_rest_h[i] = 1.0 if rh < 3.0 else 0.0
            short_rest_a[i] = 1.0 if ra < 3.0 else 0.0

            # congestion: how many games in last 7/14/30 days
            for team, arr7, arr14, arr30, idx in [
                (h, cong7_h, cong14_h, cong30_h, i),
                (a, cong7_a, cong14_a, cong30_a, i),
            ]:
                dates = match_dates.get(team, [])
                c7 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 7)
                c14 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 14)
                c30 = sum(1 for d in dates if (dt - d).total_seconds() / 86400 <= 30)
                arr7[idx] = c7; arr14[idx] = c14; arr30[idx] = c30

            # season calendar
            try:
                month = dt.month
                # season runs Aug(8) → May(5)
                if month >= 8:
                    prog = (month - 8) / 10.0  # Aug=0, Dec=0.4
                else:
                    prog = (month + 4) / 10.0  # Jan=0.5, May=0.9
                season_prog[i] = min(1.0, prog)
                is_early[i] = 1.0 if (month in (8, 9)) else 0.0
                is_late[i] = 1.0 if (month in (4, 5)) else 0.0
                d = dt.weekday()  # 0=Monday
                dow[i] = d
                is_weekend[i] = 1.0 if d >= 5 else 0.0  # Sat/Sun
                is_midweek[i] = 1.0 if d in (1, 2) else 0.0  # Tue/Wed
                hour_utc[i] = dt.hour
            except Exception:
                pass

            # context-adjusted probs
            rest_adv = (rh - ra) * 0.01
            season_adj = 0.02 if is_late[i] else 0.0  # home advantage stronger late season
            probs[i] = _norm3(0.44 + rest_adv + season_adj, 0.28, 0.28 - rest_adv - season_adj)

            # adjust confidence if short rest detected (more signal)
            if short_rest_h[i] or short_rest_a[i]:
                conf[i] = 0.45

            # --- update ---
            last_date[h] = dt; last_date[a] = dt
            match_dates.setdefault(h, []).append(dt)
            match_dates.setdefault(a, []).append(dt)

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "ctx_rest_h": rest_h, "ctx_rest_a": rest_a,
                "ctx_rest_diff": rest_h - rest_a,
                "ctx_rest_ratio": rest_ratio,
                "ctx_short_rest_h": short_rest_h, "ctx_short_rest_a": short_rest_a,
                "ctx_cong7_h": cong7_h, "ctx_cong7_a": cong7_a,
                "ctx_cong14_h": cong14_h, "ctx_cong14_a": cong14_a,
                "ctx_cong30_h": cong30_h, "ctx_cong30_a": cong30_a,
                "ctx_cong7_diff": cong7_h - cong7_a,
                "ctx_cong14_diff": cong14_h - cong14_a,
                "ctx_season_prog": season_prog,
                "ctx_is_early": is_early, "ctx_is_late": is_late,
                "ctx_dow": dow, "ctx_is_weekend": is_weekend,
                "ctx_is_midweek": is_midweek,
                "ctx_hour_utc": hour_utc,
            },
        )


# ===================================================================
# COUNCIL — META-LEARNER
# ===================================================================
ALL_EXPERTS: list[Expert] = [
    EloExpert(), MarketExpert(), FormExpert(),
    PoissonExpert(), H2HExpert(), ContextExpert(),
]


def _run_experts(df: pd.DataFrame, experts: list[Expert] | None = None) -> list[ExpertResult]:
    """Run all experts on a DataFrame of matches."""
    if experts is None:
        experts = ALL_EXPERTS
    return [expert.compute(df) for expert in experts]


def _build_meta_X(results: list[ExpertResult], experts: list[Expert] | None = None,
                  competitions: np.ndarray | None = None) -> np.ndarray:
    """Construct the meta-feature matrix from expert results.

    Layout:
        - Expert probs:          7 × 3 = 21
        - Expert confidence:     7
        - Expert domain features: ~60-70
        - Conflict signals:      ~10
        - Competition encoding:  6 (ordinal league ID)
        ≈ 110+ total columns
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
    all_probs = np.stack([r.probs for r in results], axis=0)  # (6, n, 3)
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
    mean_probs = np.mean(all_probs, axis=0)
    ens_entropy = np.array([_entropy3(p) for p in mean_probs])
    # weighted consensus (confidence-weighted mean probability)
    weights = np.stack([r.confidence for r in results], axis=0)  # (6, n)
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
    # max disagreement across any pair of experts (for home win)
    max_disagree = np.zeros(n)
    for ei in range(min(6, len(results))):
        for ej in range(ei + 1, min(6, len(results))):
            pair_dis = np.abs(results[ei].probs[:, 0] - results[ej].probs[:, 0])
            max_disagree = np.maximum(max_disagree, pair_dis)
    # number of experts that agree on same winner
    winner_votes = np.zeros(n)
    for ii in range(n):
        winners = [np.argmax(r.probs[ii]) for r in results[:6] if ii < r.probs.shape[0]]
        if winners:
            from collections import Counter
            most_common = Counter(winners).most_common(1)[0][1]
            winner_votes[ii] = most_common / len(winners)
    blocks.extend([
        elo_mkt_agree[:, None], form_h2h_agree[:, None],
        pois_mkt_agree[:, None], elo_form_agree[:, None],
        pois_elo_agree[:, None], max_disagree[:, None],
        winner_votes[:, None],
    ])

    # 5. Competition encoding (ordinal league ID)
    if competitions is not None:
        _COMP_MAP = {"PL": 1, "PD": 2, "SA": 3, "BL1": 4, "FL1": 5, "DED": 6, "ELC": 7}
        comp_ids = np.array([_COMP_MAP.get(str(c), 0) for c in competitions], dtype=float)
        blocks.append(comp_ids[:, None])

    X = np.hstack(blocks)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_df(con, finished_only: bool = True, days: int = 3650) -> pd.DataFrame:
    """Load matches + extras into a clean DataFrame for expert consumption."""
    status_filter = "m.status='FINISHED'" if finished_only else "m.status IN ('FINISHED','SCHEDULED','TIMED')"
    date_filter = f"AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL {int(days)} DAY)" if finished_only else ""

    df = con.execute(f"""
        SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
               m.home_goals, m.away_goals,
               e.b365h, e.b365d, e.b365a, e.raw_json,
               e.b365ch, e.b365cd, e.b365ca,
               e.psh, e.psd, e.psa,
               e.avgh, e.avgd, e.avga,
               e.maxh, e.maxd, e.maxa,
               e.b365_o25, e.b365_u25,
               e.avg_o25, e.avg_u25,
               e.max_o25, e.max_u25,
               e.hthg, e.htag,
               e.hs, e.hst, e.hc, e.hy, e.hr,
               e.as_ AS as_, e.ast, e.ac, e.ay, e.ar
        FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE {status_filter} {date_filter}
          AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.utc_date ASC
    """).df()

    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
        df["home_team"] = df["home_team"].map(canonical_team_name)
        df["away_team"] = df["away_team"].map(canonical_team_name)

    return df


# ---------- TRAINING ----------
def train_and_save(con, days: int = 3650, eval_days: int = 365,
                   verbose: bool = True) -> dict:
    """Train the Expert Council model."""

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
        print("[council] running 6 experts...", flush=True)
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

    # Append DC as pseudo-expert features
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

    base = HistGradientBoostingClassifier(
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
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(Xtr, ytr)

    P = model.predict_proba(Xte)
    eps = 1e-12
    logloss = float(np.mean(-np.log(P[np.arange(len(yte)), yte] + eps)))
    Y = np.zeros_like(P)
    Y[np.arange(len(yte)), yte] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
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
        "model": model,
        "dc_by_comp": dc_by_comp,
        "btts_model": btts_model,
        "ou25_model": ou25_model,
    }, MODEL_PATH)

    out = {
        "n_train": n_tr, "n_test": n_te,
        "logloss": round(logloss, 5), "brier": round(brier, 5),
        "accuracy": round(acc, 4), "ece": round(float(ece), 4),
        "n_features": int(X.shape[1]),
        "btts_accuracy": round(btts_acc, 4) if btts_model else None,
        "ou25_accuracy": round(ou25_acc, 4) if ou25_model else None,
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
    dc_by_comp = obj.get("dc_by_comp", {})

    s = settings()
    tracked = tuple(s.tracked_competitions or [])
    comp_filter, params = "", []
    if tracked:
        comp_filter = " AND m.competition IN (" + ",".join(["?"] * len(tracked)) + ")"
        params.extend(list(tracked))

    up = con.execute(f"""
        SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
               e.b365h, e.b365d, e.b365a, e.raw_json,
               e.b365ch, e.b365cd, e.b365ca,
               e.psh, e.psd, e.psa,
               e.avgh, e.avgd, e.avga,
               e.maxh, e.maxd, e.maxa,
               e.b365_o25, e.b365_u25,
               e.avg_o25, e.avg_u25,
               e.max_o25, e.max_u25
        FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status IN ('SCHEDULED','TIMED')
          AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead_days)} DAY)
          {comp_filter}
        ORDER BY m.utc_date ASC
    """, params).df()

    if up.empty:
        if verbose:
            print("[council] no upcoming matches", flush=True)
        return 0

    up["utc_date"] = pd.to_datetime(up["utc_date"], utc=True)
    up["home_team"] = up["home_team"].map(canonical_team_name)
    up["away_team"] = up["away_team"].map(canonical_team_name)

    # load all history for sequential feature computation
    hist = con.execute("""
        SELECT m.utc_date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               e.b365h, e.b365d, e.b365a, e.raw_json,
               e.b365ch, e.b365cd, e.b365ca,
               e.psh, e.psd, e.psa,
               e.avgh, e.avgd, e.avga,
               e.maxh, e.maxd, e.maxa,
               e.b365_o25, e.b365_u25,
               e.avg_o25, e.avg_u25,
               e.max_o25, e.max_u25,
               e.hs, e.hst, e.hc, e.hy, e.hr,
               e.as_ AS as_, e.ast, e.ac, e.ay, e.ar
        FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status = 'FINISHED'
        ORDER BY m.utc_date ASC
    """).df()
    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)
    hist["home_team"] = hist["home_team"].map(canonical_team_name)
    hist["away_team"] = hist["away_team"].map(canonical_team_name)

    # dummy rows for upcoming
    dummy = up[["utc_date", "home_team", "away_team", "b365h", "b365d", "b365a", "raw_json"]].copy()
    dummy["home_goals"] = 0; dummy["away_goals"] = 0
    for c in ["hs", "hst", "hc", "hy", "hr", "as_", "ast", "ac", "ay", "ar"]:
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

    # for upcoming only, compute DC from stored model
    if "competition" not in combo.columns:
        # add competition from up dataframe
        combo_comp = pd.concat([
            hist[["utc_date"]].assign(competition=""),
            up[["utc_date", "competition"]],
        ], ignore_index=True).sort_values("utc_date").reset_index(drop=True)
    else:
        combo_comp = combo

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
    P = model.predict_proba(X)

    # BTTS & O2.5 model heads
    btts_model = obj.get("btts_model")
    ou25_model = obj.get("ou25_model")
    P_btts = btts_model.predict_proba(X) if btts_model else None
    P_ou25 = ou25_model.predict_proba(X) if ou25_model else None

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

        # Use trained model heads if available, else fall back to Poisson
        btts_val = float(P_btts[j, 1]) if P_btts is not None else btts_pois
        o25_val = float(P_ou25[j, 1]) if P_ou25 is not None else o25_pois

        notes_dict = {
            "model": "Expert Council (6 experts + meta-learner)",
            "btts": round(btts_val, 3),
            "o25": round(o25_val, 3),
            "btts_poisson": round(btts_pois, 3),
            "o25_poisson": round(o25_pois, 3),
            "predicted_score": [ml_hg, ml_ag],
            "lambda_home": round(lam_h_val, 2),
            "lambda_away": round(lam_a_val, 2),
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
    hist = con.execute("""
        SELECT m.utc_date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               e.b365h, e.b365d, e.b365a, e.raw_json,
               e.hs, e.hst, e.hc, e.hy, e.hr,
               e.as_ AS as_, e.ast, e.ac, e.ay, e.ar
        FROM matches m
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status = 'FINISHED'
          AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL 5 YEAR)
        ORDER BY m.utc_date ASC
    """).df()

    if hist.empty:
        return None

    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)
    hist["home_team"] = hist["home_team"].map(canonical_team_name)
    hist["away_team"] = hist["away_team"].map(canonical_team_name)

    # Build dummy row for this match
    extras = con.execute(
        "SELECT b365h, b365d, b365a, raw_json FROM match_extras WHERE match_id = ?",
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
        "hs": 0, "hst": 0, "hc": 0, "hy": 0, "hr": 0,
        "as_": 0, "ast": 0, "ac": 0, "ay": 0, "ar": 0,
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
