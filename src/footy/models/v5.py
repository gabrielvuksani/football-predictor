from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from footy.normalize import canonical_team_name
from footy.config import settings
from footy.models.dixon_coles import fit_dc, predict_1x2, DCModel

V5_MODEL_VERSION = "v5_ultimate"
MODEL_PATH = Path("data/models") / f"{V5_MODEL_VERSION}.joblib"

def _label(hg: int, ag: int) -> int:
    if hg > ag: return 0
    if hg == ag: return 1
    return 2

def _to_float(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        v = float(x)
        if not np.isfinite(v): return None
        return v
    except Exception:
        return None

def _raw_dict(raw_json):
    if raw_json is None:
        return {}
    try:
        if isinstance(raw_json, dict):
            return raw_json
        return json.loads(raw_json)
    except Exception:
        return {}

def implied_1x2(h, d, a):
    h = _to_float(h); d = _to_float(d); a = _to_float(a)
    if not h or not d or not a:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    ih, id_, ia = 1.0/h, 1.0/d, 1.0/a
    s = ih + id_ + ia
    return (float(ih/s), float(id_/s), float(ia/s), 1.0, float(s - 1.0))

def market_probs(row_b365h, row_b365d, row_b365a, raw_json):
    """
    Prefer closing -> Avg -> Max -> open.
    closing odds naming is documented by football-data.co.uk notes.txt. :contentReference[oaicite:2]{index=2}
    """
    r = _raw_dict(raw_json)
    for keys in [("B365CH","B365CD","B365CA"), ("AvgH","AvgD","AvgA"), ("MaxH","MaxD","MaxA"), ("B365H","B365D","B365A")]:
        h = r.get(keys[0], row_b365h)
        d = r.get(keys[1], row_b365d)
        a = r.get(keys[2], row_b365a)
        ph, pd, pa, has, over = implied_1x2(h, d, a)
        if has > 0:
            src = 3.0 if keys[0].endswith("CH") else (2.0 if keys[0].startswith("Avg") else (1.0 if keys[0].startswith("Max") else 0.0))
            return ph, pd, pa, has, over, src
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def odds_movement(raw_json):
    """
    Movement = closing implied - opening implied (Bet365).
    """
    r = _raw_dict(raw_json)
    o = implied_1x2(r.get("B365H"), r.get("B365D"), r.get("B365A"))
    c = implied_1x2(r.get("B365CH"), r.get("B365CD"), r.get("B365CA"))
    if o[3] == 0.0 or c[3] == 0.0:
        return (0.0, 0.0, 0.0, 0.0)  # dH,dD,dA,has_move
    return (float(c[0]-o[0]), float(c[1]-o[1]), float(c[2]-o[2]), 1.0)

def ou25_probs(raw_json):
    r = _raw_dict(raw_json)
    candidates = [
        ("AvgC>2.5","AvgC<2.5"),
        ("Avg>2.5","Avg<2.5"),
        ("B365C>2.5","B365C<2.5"),
        ("B365>2.5","B365<2.5"),
    ]
    for o_key, u_key in candidates:
        o = _to_float(r.get(o_key))
        u = _to_float(r.get(u_key))
        if o and u:
            io, iu = 1.0/o, 1.0/u
            s = io + iu
            p_over = io / s
            return float(p_over), float(s - 1.0), 1.0
    return 0.0, 0.0, 0.0

def entropy3(p):
    p = np.clip(np.array(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def disagreement(ps):
    ps = np.array(ps, dtype=float)
    return float(np.mean(np.std(ps, axis=0)))

def _valid_prob_row(row: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(row)) and np.all(row >= 0) and float(row.sum()) > 0)

def _normalize_prob_row(row: np.ndarray) -> np.ndarray:
    s = float(row.sum())
    if s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=float)
    return row / s

def _repair_prob_matrix(P_raw: np.ndarray, fallback_raw: np.ndarray) -> np.ndarray:
    P_out = np.asarray(P_raw, dtype=float).copy()
    fallback = np.asarray(fallback_raw, dtype=float).copy()

    for i in range(len(fallback)):
        if _valid_prob_row(fallback[i]):
            fallback[i] = _normalize_prob_row(fallback[i])
        else:
            fallback[i] = np.array([1/3, 1/3, 1/3], dtype=float)

    for i in range(len(P_out)):
        row = P_out[i]
        if _valid_prob_row(row):
            P_out[i] = _normalize_prob_row(row)
        else:
            P_out[i] = fallback[i]
    return P_out

# ---------- sequential: H2H + rest + rolling stats + Elo + Poisson ----------
def build_seq_features(df: pd.DataFrame, n_h2h: int = 6, n_venue: int = 3, n_roll: int = 5,
                       h2h_decay_half_life_days: float = 730.0):
    """
    df sorted by utc_date asc.
    Returns arrays (all same length as df) for a large set of pre-match features.
    Computes sequential Elo, Poisson-like strengths, team-specific home advantage,
    and time-decayed H2H — all leak-free.
    """
    import math as _math

    last_date = {}
    h2h = {}     # frozenset(home,away) -> list of (home, away, hg, ag, utc_date)
    venue = {}

    # rolling per team
    roll = {}  # team -> list of dicts

    # Sequential Elo state
    elo_ratings = {}   # team -> float
    elo_match_counts = {}  # team -> int
    ELO_DEFAULT = 1500.0
    ELO_K_BASE = 20.0
    ELO_BLANKET_HOME = 40.0  # reduced blanket; team-specific adds on top

    # Team-specific home advantage tracking
    home_record = {}  # team -> deque-like list of (pts_at_home, gf, ga)
    away_record = {}  # team -> deque-like list of (pts_away, gf, ga)
    HOME_TRACK_N = 20  # how many home/away games to track

    # Sequential Poisson-like strength tracking (exponentially weighted)
    team_attack = {}   # team -> running avg goals scored
    team_defense = {}  # team -> running avg goals conceded
    POISSON_ALPHA = 0.08  # EMA smoothing factor

    rest_h, rest_a = [], []
    h2h_n, h2h_ph, h2h_pa, h2h_gd = [], [], [], []
    v_n, v_ph, v_gd = [], [], []

    # rolling outputs
    rf_h, ra_h, rp_h = [], [], []   # goals_for_avg, goals_against_avg, points_avg
    rf_a, ra_a, rp_a = [], [], []
    sh_h, sot_h, cor_h, card_h = [], [], [], []
    sh_a, sot_a, cor_a, card_a = [], [], [], []

    # Extended features
    cs_h, cs_a = [], []             # clean sheet %
    btts_h, btts_a = [], []         # both teams to score %
    momentum_h, momentum_a = [], [] # form momentum (recent vs older)
    streak_h, streak_a = [], []     # win streak (positive) / loss streak (negative)
    gsup_h, gsup_a = [], []         # avg goal supremacy (gf - ga)
    sot_ratio_h, sot_ratio_a = [], [] # shots on target ratio (sot/shots)

    # Sequential Elo outputs (leak-free pre-match values)
    seq_elo_h, seq_elo_a = [], []
    seq_elo_ph, seq_elo_pd, seq_elo_pa = [], [], []

    # Sequential Poisson-like outputs
    seq_poisson_lam_h, seq_poisson_lam_a = [], []

    # Team-specific home advantage outputs
    seq_home_strength_h, seq_away_strength_a = [], []

    def _points(gf, ga):
        return 3 if gf > ga else (1 if gf == ga else 0)

    def _avg(team, key):
        xs = [d.get(key, 0.0) for d in roll.get(team, [])[-n_roll:]]
        return float(np.mean(xs)) if xs else 0.0

    def _rate(team, key, window=10):
        xs = [d.get(key, 0.0) for d in roll.get(team, [])[-window:]]
        return float(np.mean(xs)) if xs else 0.0

    def _momentum(team, n_recent=3, n_older=5):
        history = roll.get(team, [])
        if len(history) < 2:
            return 0.0
        recent = [d["pts"] for d in history[-n_recent:]]
        older = [d["pts"] for d in history[-n_older:]]
        return float(np.mean(recent) - np.mean(older)) if older else 0.0

    def _streak(team):
        history = roll.get(team, [])
        if not history:
            return 0.0
        s = 0
        last_result = history[-1]["pts"]
        for d in reversed(history):
            if last_result == 3 and d["pts"] == 3:
                s += 1
            elif last_result == 0 and d["pts"] == 0:
                s -= 1
            else:
                break
        return float(s)

    def _get_elo(team):
        return elo_ratings.get(team, ELO_DEFAULT)

    def _elo_expected(r_home, r_away):
        return 1.0 / (1.0 + 10.0 ** (-(r_home - r_away) / 400.0))

    def _elo_dynamic_k(team, goal_diff):
        n = elo_match_counts.get(team, 0)
        convergence = 1.0 + max(0.0, 1.0 - n / 60.0)
        gd = abs(goal_diff)
        gd_mult = 1.0 if gd <= 1 else (1.15 if gd == 2 else (1.30 if gd == 3 else min(1.50, 1.30 + 0.05 * (gd - 3))))
        return ELO_K_BASE * convergence * gd_mult

    def _team_home_advantage(team):
        """Compute team-specific home advantage from tracked home/away records."""
        h_rec = home_record.get(team, [])
        a_rec = away_record.get(team, [])
        if len(h_rec) < 3 or len(a_rec) < 3:
            return ELO_BLANKET_HOME  # not enough data, use blanket
        # Home PPG vs Away PPG
        h_ppg = np.mean([r[0] for r in h_rec[-HOME_TRACK_N:]])
        a_ppg = np.mean([r[0] for r in a_rec[-HOME_TRACK_N:]])
        # Home GD vs Away GD
        h_gd = np.mean([r[1] - r[2] for r in h_rec[-HOME_TRACK_N:]])
        a_gd = np.mean([r[1] - r[2] for r in a_rec[-HOME_TRACK_N:]])
        # Team-specific home advantage: higher when team performs much better at home
        ppg_diff = h_ppg - a_ppg  # typically 0.0 to 1.0 for strong home teams
        gd_diff = h_gd - a_gd    # goal difference diff
        # Scale to Elo points: baseline 40, adjusted by performance
        # ppg_diff of 0.5 ≈ +25 Elo, 1.0 ≈ +50 Elo
        adv = ELO_BLANKET_HOME + ppg_diff * 50.0 + gd_diff * 10.0
        return max(0.0, min(150.0, adv))  # clamp to reasonable range

    def _home_strength_feature(team):
        """Normalized home strength: >0 = better at home, <0 = worse at home."""
        h_rec = home_record.get(team, [])
        a_rec = away_record.get(team, [])
        if len(h_rec) < 3 or len(a_rec) < 3:
            return 0.0
        h_ppg = np.mean([r[0] for r in h_rec[-HOME_TRACK_N:]])
        a_ppg = np.mean([r[0] for r in a_rec[-HOME_TRACK_N:]])
        return float(h_ppg - a_ppg)

    def _away_strength_feature(team):
        """Normalized away strength: >0 = good away form."""
        a_rec = away_record.get(team, [])
        if len(a_rec) < 3:
            return 0.0
        return float(np.mean([r[0] for r in a_rec[-HOME_TRACK_N:]]) / 3.0)

    def _get_attack(team):
        return team_attack.get(team, 1.3)

    def _get_defense(team):
        return team_defense.get(team, 1.2)

    LEAGUE_AVG_GOALS = 1.35  # approximate average goal rate

    for r in df.itertuples(index=False):
        dt = r.utc_date
        home = canonical_team_name(r.home_team)
        away = canonical_team_name(r.away_team)

        # rest days
        lh = last_date.get(home)
        la = last_date.get(away)
        rh = 7.0 if lh is None else max(0.0, (dt - lh).total_seconds()/86400.0)
        ra = 7.0 if la is None else max(0.0, (dt - la).total_seconds()/86400.0)
        rest_h.append(float(rh)); rest_a.append(float(ra))

        # ---- Sequential Elo (PRE-match values) ----
        elo_home_raw = _get_elo(home)
        elo_away_raw = _get_elo(away)
        home_adv_elo = _team_home_advantage(home)
        elo_home_adj = elo_home_raw + home_adv_elo
        elo_away_adj = elo_away_raw

        seq_elo_h.append(float(elo_home_raw))
        seq_elo_a.append(float(elo_away_raw))

        # Elo probs
        e_home = _elo_expected(elo_home_adj, elo_away_adj)
        rating_diff = abs(elo_home_adj - elo_away_adj)
        p_draw_elo = 0.26 + 0.06 * _math.exp(-rating_diff / 200.0)
        p_draw_elo = max(0.18, min(0.34, p_draw_elo))
        p_home_elo = e_home * (1.0 - p_draw_elo)
        p_away_elo = (1.0 - e_home) * (1.0 - p_draw_elo)
        s_elo = p_home_elo + p_draw_elo + p_away_elo
        seq_elo_ph.append(float(p_home_elo / s_elo))
        seq_elo_pd.append(float(p_draw_elo / s_elo))
        seq_elo_pa.append(float(p_away_elo / s_elo))

        # ---- Sequential Poisson-like (PRE-match values) ----
        atk_h = _get_attack(home)
        def_h = _get_defense(home)
        atk_a = _get_attack(away)
        def_a = _get_defense(away)
        # Expected goals: attack of scoring team vs defense of conceding team
        lam_h = max(0.2, min(5.0, atk_h * def_a / LEAGUE_AVG_GOALS))
        lam_a = max(0.2, min(5.0, atk_a * def_h / LEAGUE_AVG_GOALS))
        seq_poisson_lam_h.append(float(lam_h))
        seq_poisson_lam_a.append(float(lam_a))

        # ---- Team-specific home advantage features ----
        seq_home_strength_h.append(_home_strength_feature(home))
        seq_away_strength_a.append(_away_strength_feature(away))

        # ---- H2H (with time-decay) ----
        key = frozenset([home, away])
        past_all = h2h.get(key, [])
        # Apply time-decay weighting
        decay_lambda = np.log(2.0) / h2h_decay_half_life_days
        n_weighted = 0.0
        pts_home_w = pts_away_w = gd_home_w = 0.0
        for phome, paway, hg_past, ag_past, past_dt in past_all[-(n_h2h * 2):]:
            # time decay weight
            age_days = max(0.0, (dt - past_dt).total_seconds() / 86400.0)
            w = _math.exp(-decay_lambda * age_days)
            if phome == home and paway == away:
                pts_home_w += _points(hg_past, ag_past) * w
                pts_away_w += _points(ag_past, hg_past) * w
                gd_home_w += (hg_past - ag_past) * w
            else:
                pts_home_w += _points(ag_past, hg_past) * w
                pts_away_w += _points(hg_past, ag_past) * w
                gd_home_w += (ag_past - hg_past) * w
            n_weighted += w

        n_raw = len(past_all[-(n_h2h * 2):])
        h2h_n.append(float(n_raw))
        h2h_ph.append(float(pts_home_w / max(1e-6, n_weighted)))
        h2h_pa.append(float(pts_away_w / max(1e-6, n_weighted)))
        h2h_gd.append(float(gd_home_w / max(1e-6, n_weighted)))

        # Venue H2H (same home/away pairing)
        vpast = venue.get((home, away), [])[-n_venue:]
        vn = len(vpast)
        vpts = vgd = 0.0
        for hg_v, ag_v in vpast:
            vpts += _points(hg_v, ag_v)
            vgd += (hg_v - ag_v)
        v_n.append(float(vn))
        v_ph.append(float(vpts / max(1, vn)))
        v_gd.append(float(vgd / max(1, vn)))

        # rolling (pre-match)
        rf_h.append(_avg(home, "gf"))
        ra_h.append(_avg(home, "ga"))
        rp_h.append(_avg(home, "pts"))
        rf_a.append(_avg(away, "gf"))
        ra_a.append(_avg(away, "ga"))
        rp_a.append(_avg(away, "pts"))

        sh_h.append(_avg(home, "sh"))
        sot_h.append(_avg(home, "sot"))
        cor_h.append(_avg(home, "cor"))
        card_h.append(_avg(home, "card"))

        sh_a.append(_avg(away, "sh"))
        sot_a.append(_avg(away, "sot"))
        cor_a.append(_avg(away, "cor"))
        card_a.append(_avg(away, "card"))

        cs_h.append(_rate(home, "cs"))
        cs_a.append(_rate(away, "cs"))
        btts_h.append(_rate(home, "btts"))
        btts_a.append(_rate(away, "btts"))
        momentum_h.append(_momentum(home))
        momentum_a.append(_momentum(away))
        streak_h.append(_streak(home))
        streak_a.append(_streak(away))
        gsup_h.append(_avg(home, "gsup"))
        gsup_a.append(_avg(away, "gsup"))

        sh_val_h = _avg(home, "sh")
        sot_val_h = _avg(home, "sot")
        sot_ratio_h.append(sot_val_h / max(sh_val_h, 0.1))
        sh_val_a = _avg(away, "sh")
        sot_val_a = _avg(away, "sot")
        sot_ratio_a.append(sot_val_a / max(sh_val_a, 0.1))

        # ======== UPDATE STATE AFTER RECORDING PRE-MATCH VALUES ========
        last_date[home] = dt
        last_date[away] = dt

        hg = int(r.home_goals); ag = int(r.away_goals)

        # H2H state (now with date)
        h2h.setdefault(key, []).append((home, away, hg, ag, dt))
        venue.setdefault((home, away), []).append((hg, ag))

        # Elo update (post-match)
        goal_diff = hg - ag
        s_home_elo = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        exp_home = _elo_expected(elo_home_adj, elo_away_adj)
        k_home = _elo_dynamic_k(home, goal_diff)
        k_away = _elo_dynamic_k(away, -goal_diff)
        k_avg = (k_home + k_away) / 2.0
        delta = k_avg * (s_home_elo - exp_home)
        elo_ratings[home] = elo_home_raw + delta
        elo_ratings[away] = elo_away_raw - delta
        elo_match_counts[home] = elo_match_counts.get(home, 0) + 1
        elo_match_counts[away] = elo_match_counts.get(away, 0) + 1

        # Poisson-like EMA update (post-match)
        team_attack[home] = (1.0 - POISSON_ALPHA) * _get_attack(home) + POISSON_ALPHA * hg if home in team_attack else float(hg) if hg > 0 else 1.3
        team_defense[home] = (1.0 - POISSON_ALPHA) * _get_defense(home) + POISSON_ALPHA * ag if home in team_defense else float(ag) if ag > 0 else 1.2
        team_attack[away] = (1.0 - POISSON_ALPHA) * _get_attack(away) + POISSON_ALPHA * ag if away in team_attack else float(ag) if ag > 0 else 1.3
        team_defense[away] = (1.0 - POISSON_ALPHA) * _get_defense(away) + POISSON_ALPHA * hg if away in team_defense else float(hg) if hg > 0 else 1.2

        # Home/away record tracking (post-match)
        home_record.setdefault(home, []).append((_points(hg, ag), hg, ag))
        away_record.setdefault(away, []).append((_points(ag, hg), ag, hg))

        # rolling state (shots/corners/cards from extras if available)
        hs = float(getattr(r, "hs", 0.0) or 0.0)
        hst = float(getattr(r, "hst", 0.0) or 0.0)
        hc = float(getattr(r, "hc", 0.0) or 0.0)
        hy = float(getattr(r, "hy", 0.0) or 0.0)
        hr = float(getattr(r, "hr", 0.0) or 0.0)

        as_ = float(getattr(r, "as_", 0.0) or 0.0)
        ast = float(getattr(r, "ast", 0.0) or 0.0)
        ac = float(getattr(r, "ac", 0.0) or 0.0)
        ay = float(getattr(r, "ay", 0.0) or 0.0)
        ar = float(getattr(r, "ar", 0.0) or 0.0)

        # simple card "points"
        hcard = hy + 2.5 * hr
        acard = ay + 2.5 * ar

        roll.setdefault(home, []).append({
            "gf": hg, "ga": ag, "pts": _points(hg, ag),
            "sh": hs, "sot": hst, "cor": hc, "card": hcard,
            "cs": 1.0 if ag == 0 else 0.0,
            "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
            "gsup": hg - ag,
        })
        roll.setdefault(away, []).append({
            "gf": ag, "ga": hg, "pts": _points(ag, hg),
            "sh": as_, "sot": ast, "cor": ac, "card": acard,
            "cs": 1.0 if hg == 0 else 0.0,
            "btts": 1.0 if (hg > 0 and ag > 0) else 0.0,
            "gsup": ag - hg,
        })

    return (
        np.array(rest_h), np.array(rest_a),
        np.array(h2h_n), np.array(h2h_ph), np.array(h2h_pa), np.array(h2h_gd),
        np.array(v_n), np.array(v_ph), np.array(v_gd),
        np.array(rf_h), np.array(ra_h), np.array(rp_h),
        np.array(rf_a), np.array(ra_a), np.array(rp_a),
        np.array(sh_h), np.array(sot_h), np.array(cor_h), np.array(card_h),
        np.array(sh_a), np.array(sot_a), np.array(cor_a), np.array(card_a),
        # extended
        np.array(cs_h), np.array(cs_a),
        np.array(btts_h), np.array(btts_a),
        np.array(momentum_h), np.array(momentum_a),
        np.array(streak_h), np.array(streak_a),
        np.array(gsup_h), np.array(gsup_a),
        np.array(sot_ratio_h), np.array(sot_ratio_a),
        # sequential Elo (leak-free)
        np.array(seq_elo_h), np.array(seq_elo_a),
        np.array(seq_elo_ph), np.array(seq_elo_pd), np.array(seq_elo_pa),
        # sequential Poisson-like
        np.array(seq_poisson_lam_h), np.array(seq_poisson_lam_a),
        # team-specific home advantage
        np.array(seq_home_strength_h), np.array(seq_away_strength_a),
    )

def h2h_expert_probs(n, pts_h, pts_a, gd_h):
    """
    Convert H2H summary into a soft expert distribution.
    The model will learn when to trust this expert via features.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0, 0.0)  # ph,pd,pa,strength
    # strength increases with sample size
    strength = min(1.0, n / 6.0)
    # score = points diff + small gd component
    s = (pts_h - pts_a) + 0.25 * gd_h
    # draw base higher when close
    ph = 1.0 / (1.0 + np.exp(-1.2 * s))
    pa = 1.0 - ph
    pd = max(0.10, 0.28 - abs(s) * 0.06)
    # renorm
    ph = ph * (1.0 - pd)
    pa = pa * (1.0 - pd)
    z = ph + pd + pa
    return (float(ph/z), float(pd/z), float(pa/z), float(strength))

# ---------- training ----------
def train_and_save(con, days: int = 3650, eval_days: int = 365, verbose: bool = True) -> dict:
    # No longer requires walk-forward predictions — all features computed inline
    df = con.execute(
        f"""SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
                   m.home_goals, m.away_goals,
                   e.b365h, e.b365d, e.b365a, e.raw_json,
                   e.hs, e.hst, e.hc, e.hy, e.hr,
                   e.as_ AS as_, e.ast, e.ac, e.ay, e.ar
            FROM matches m
            LEFT JOIN match_extras e ON e.match_id=m.match_id
            WHERE m.status='FINISHED'
              AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
              AND m.utc_date >= (CURRENT_TIMESTAMP - INTERVAL {int(days)} DAY)
            ORDER BY m.utc_date ASC"""
    ).df()

    if df.empty or len(df) < 800:
        return {"error": f"Not enough data to train v5 (got {len(df)}). Ingest more history."}

    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    cutoff = df["utc_date"].max() - pd.Timedelta(days=int(eval_days))
    train_mask = df["utc_date"] < cutoff
    test_mask = ~train_mask

    n_tr = int(train_mask.sum()); n_te = int(test_mask.sum())
    if verbose:
        print(f"[v5] total={len(df)} train={n_tr} test={n_te} cutoff={cutoff}", flush=True)
    if n_tr < 500:
        return {"error": f"Train split too small (train={n_tr}). Ingest more history."}

    # canonicalize team names
    df["home_team"] = df["home_team"].map(canonical_team_name)
    df["away_team"] = df["away_team"].map(canonical_team_name)

    # sequential features — all leak-free (Elo, Poisson, H2H, form, etc.)
    seq_cols = ["utc_date","home_team","away_team","home_goals","away_goals",
                "hs","hst","hc","hy","hr","as_","ast","ac","ay","ar"]
    seq = df[seq_cols].copy()
    (rest_h, rest_a,
     h2n, h2ph, h2pa, h2gd,
     vn, vph, vgd,
     gf_h, ga_h, pts_h,
     gf_a, ga_a, pts_a,
     sh_h, sot_h, cor_h, card_h,
     sh_a, sot_a, cor_a, card_a,
     cs_h, cs_a, btts_h, btts_a,
     momentum_h, momentum_a, streak_h, streak_a,
     gsup_h, gsup_a, sot_ratio_h, sot_ratio_a,
     elo_h, elo_a,
     elo_ph, elo_pd, elo_pa,
     poisson_lam_h, poisson_lam_a,
     home_strength_h, away_strength_a,
    ) = build_seq_features(seq)

    # market odds (closing/avg/max) + movement + ou25
    mk = df.apply(lambda r: market_probs(r.b365h, r.b365d, r.b365a, r.raw_json), axis=1, result_type="expand")
    mk.columns = ["m_h","m_d","m_a","has_m","m_over","m_src"]

    mv = df["raw_json"].apply(odds_movement).apply(pd.Series)
    mv.columns = ["mv_h","mv_d","mv_a","has_mv"]

    ou = df["raw_json"].apply(ou25_probs).apply(pd.Series)
    ou.columns = ["p_over25_mkt","ou_over","has_ou"]

    PM = mk[["m_h","m_d","m_a"]].to_numpy(float)
    PM = _repair_prob_matrix(PM, PM)

    # Elo expert probs (from sequential computation)
    P_elo = np.column_stack([elo_ph, elo_pd, elo_pa])

    # H2H expert
    H2 = np.array([h2h_expert_probs(n, ph, pa, gd) for n, ph, pa, gd in zip(h2n, h2ph, h2pa, h2gd)], dtype=float)
    PH2 = H2[:, :3]
    H2S = H2[:, 3:4]  # strength

    # Dixon–Coles expert fitted ONLY on train split, per competition
    df_train = df[train_mask].copy()
    dc_by_comp: dict[str, DCModel] = {}
    for comp, g in df_train.groupby("competition"):
        m = fit_dc(g[["utc_date","home_team","away_team","home_goals","away_goals"]], half_life_days=365.0, max_goals=10)
        if m is not None:
            dc_by_comp[str(comp)] = m

    dc_p = np.zeros((len(df), 3), dtype=float)
    dc_eg = np.zeros((len(df), 2), dtype=float)
    dc_o25 = np.zeros((len(df), 1), dtype=float)
    has_dc = np.zeros((len(df), 1), dtype=float)

    for i, r in enumerate(df[["competition","home_team","away_team"]].itertuples(index=False)):
        comp = str(r.competition)
        m = dc_by_comp.get(comp)
        if m is None:
            continue
        ph, p_draw, pa, eg_h, eg_a, p_over = predict_1x2(m, r.home_team, r.away_team)
        if ph > 0:
            dc_p[i] = [ph, p_draw, pa]
            dc_eg[i] = [eg_h, eg_a]
            dc_o25[i] = [p_over]
            has_dc[i] = [1.0]

    # Entropy and disagreement across experts
    ent_elo = np.array([entropy3(p) for p in P_elo])
    entm = np.array([entropy3(p) for p in PM])
    entdc = np.array([entropy3(p) for p in dc_p])
    enth2h = np.array([entropy3(p) for p in PH2])
    dis = np.array([disagreement(np.vstack([a,b,c])) for a,b,c in zip(P_elo, PM, dc_p)])

    # Feature matrix — no walk-forward dependency, all computed inline
    X = np.column_stack([
        # Expert probs: Elo (3) + Market (6) + H2H (4) + DC (7) = 20
        P_elo,
        PM, mk[["has_m","m_over","m_src"]].to_numpy(float),
        PH2, H2S,
        dc_p, has_dc, dc_eg, dc_o25,
        # Odds features: movement (4) + ou (3) = 7
        mv.to_numpy(float),
        ou.to_numpy(float),
        # Expert agreement: entropy (4) + disagreement (1) = 5
        ent_elo, entm, entdc, enth2h, dis,
        # Rest days (3)
        rest_h, rest_a, (rest_h - rest_a),
        # H2H numeric (7)
        h2n, h2ph, h2pa, h2gd, vn, vph, vgd,
        # Rolling form (14)
        gf_h, ga_h, pts_h, gf_a, ga_a, pts_a,
        sh_h, sot_h, cor_h, card_h,
        sh_a, sot_a, cor_a, card_a,
        # Extended form (12)
        cs_h, cs_a, btts_h, btts_a,
        momentum_h, momentum_a, streak_h, streak_a,
        gsup_h, gsup_a, sot_ratio_h, sot_ratio_a,
        # Sequential Elo ratings + diff (3)
        elo_h, elo_a, (elo_h - elo_a),
        # Sequential Poisson + derived (5)
        poisson_lam_h, poisson_lam_a,
        (poisson_lam_h - poisson_lam_a),
        (poisson_lam_h + poisson_lam_a),
        (poisson_lam_h * poisson_lam_a),
        # Team-specific home advantage (3)
        home_strength_h, away_strength_a, (home_strength_h - away_strength_a),
        # Form diffs (7)
        (gf_h - gf_a), (ga_h - ga_a), (pts_h - pts_a),
        (sh_h - sh_a), (sot_h - sot_a), (cor_h - cor_a), (card_h - card_a),
        # Extended diffs (4)
        (cs_h - cs_a), (btts_h - btts_a), (momentum_h - momentum_a), (gsup_h - gsup_a),
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.array([_label(int(hg), int(ag)) for hg, ag in zip(df["home_goals"], df["away_goals"])], dtype=int)

    Xtr, ytr = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    Xte, yte = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    base = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=7,
        max_iter=1100,
        l2_regularization=0.30,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(Xtr, ytr)

    P = model.predict_proba(Xte)

    # No market anchor blend — the model learns proper weighting through its features
    eps = 1e-12
    logloss = float(np.mean(-np.log(P[np.arange(len(yte)), yte] + eps)))
    Y = np.zeros_like(P); Y[np.arange(len(yte)), yte] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(P, axis=1) == yte))

    # ECE (10 bin)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == yte).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.any():
            ece += abs(conf[m].mean() - correct[m].mean()) * (m.mean())
    ece = float(ece)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "dc_by_comp": dc_by_comp,
        },
        MODEL_PATH
    )

    out = {"n_test": int(len(yte)), "logloss": logloss, "brier": brier, "accuracy": acc, "ece": ece}
    if verbose:
        print(f"[v5] saved {V5_MODEL_VERSION} to {MODEL_PATH} | {out}", flush=True)
    return out

def load():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

# ---------- prediction ----------
def predict_upcoming(con, lookahead_days: int = 7, verbose: bool = True) -> int:
    obj = load()
    if obj is None:
        raise RuntimeError("v5 model not found. Run: footy train-v5")
    model = obj["model"]
    dc_by_comp = obj.get("dc_by_comp", {})

    s = settings()
    tracked = tuple(s.tracked_competitions or [])
    comp_filter = ""
    params = []
    if tracked:
        comp_filter = " AND m.competition IN (" + ",".join(["?"] * len(tracked)) + ")"
        params.extend(list(tracked))

    # No longer requires v1/v2/v3 predictions
    up = con.execute(
        f"""SELECT m.match_id, m.utc_date, m.competition, m.home_team, m.away_team,
                   e.b365h, e.b365d, e.b365a, e.raw_json
            FROM matches m
            LEFT JOIN match_extras e ON e.match_id=m.match_id
            WHERE m.status IN ('SCHEDULED','TIMED')
              AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead_days)} DAY)
              {comp_filter}
            ORDER BY m.utc_date ASC""",
        params
    ).df()

    if up.empty:
        if verbose: print("[v5] no upcoming matches", flush=True)
        return 0

    up["utc_date"] = pd.to_datetime(up["utc_date"], utc=True)
    up["home_team"] = up["home_team"].map(canonical_team_name)
    up["away_team"] = up["away_team"].map(canonical_team_name)

    # Build sequential features by processing all history + upcoming dummies
    hist = con.execute(
        """SELECT utc_date, home_team, away_team, home_goals, away_goals,
                  e.hs, e.hst, e.hc, e.hy, e.hr,
                  e.as_ AS as_, e.ast, e.ac, e.ay, e.ar
           FROM matches m
           LEFT JOIN match_extras e ON e.match_id=m.match_id
           WHERE m.status='FINISHED'
           ORDER BY utc_date ASC"""
    ).df()
    hist["utc_date"] = pd.to_datetime(hist["utc_date"], utc=True)
    hist["home_team"] = hist["home_team"].map(canonical_team_name)
    hist["away_team"] = hist["away_team"].map(canonical_team_name)

    # append dummy rows for upcoming to compute pre-match seq features
    dummy = up[["utc_date","home_team","away_team"]].copy()
    dummy["home_goals"] = 0
    dummy["away_goals"] = 0
    for c in ["hs","hst","hc","hy","hr","as_","ast","ac","ay","ar"]:
        dummy[c] = 0

    combo = pd.concat([hist, dummy], ignore_index=True).sort_values("utc_date").reset_index(drop=True)
    (rest_h, rest_a,
     h2n, h2ph, h2pa, h2gd,
     vn, vph, vgd,
     gf_h, ga_h, pts_h,
     gf_a, ga_a, pts_a,
     sh_h, sot_h, cor_h, card_h,
     sh_a, sot_a, cor_a, card_a,
     cs_h, cs_a, btts_h, btts_a,
     momentum_h, momentum_a, streak_h, streak_a,
     gsup_h, gsup_a, sot_ratio_h, sot_ratio_a,
     elo_h, elo_a,
     elo_ph, elo_pd, elo_pa,
     poisson_lam_h, poisson_lam_a,
     home_strength_h, away_strength_a,
    ) = build_seq_features(combo)

    # take only last len(up) rows (the upcoming dummy rows)
    tail = len(up)
    rest_h = rest_h[-tail:]; rest_a = rest_a[-tail:]
    h2n = h2n[-tail:]; h2ph = h2ph[-tail:]; h2pa = h2pa[-tail:]; h2gd = h2gd[-tail:]
    vn = vn[-tail:]; vph = vph[-tail:]; vgd = vgd[-tail:]
    gf_h = gf_h[-tail:]; ga_h = ga_h[-tail:]; pts_h = pts_h[-tail:]
    gf_a = gf_a[-tail:]; ga_a = ga_a[-tail:]; pts_a = pts_a[-tail:]
    sh_h = sh_h[-tail:]; sot_h = sot_h[-tail:]; cor_h = cor_h[-tail:]; card_h = card_h[-tail:]
    sh_a = sh_a[-tail:]; sot_a = sot_a[-tail:]; cor_a = cor_a[-tail:]; card_a = card_a[-tail:]
    cs_h = cs_h[-tail:]; cs_a = cs_a[-tail:]
    btts_h = btts_h[-tail:]; btts_a = btts_a[-tail:]
    momentum_h = momentum_h[-tail:]; momentum_a = momentum_a[-tail:]
    streak_h = streak_h[-tail:]; streak_a = streak_a[-tail:]
    gsup_h = gsup_h[-tail:]; gsup_a = gsup_a[-tail:]
    sot_ratio_h = sot_ratio_h[-tail:]; sot_ratio_a = sot_ratio_a[-tail:]
    elo_h = elo_h[-tail:]; elo_a = elo_a[-tail:]
    elo_ph = elo_ph[-tail:]; elo_pd = elo_pd[-tail:]; elo_pa = elo_pa[-tail:]
    poisson_lam_h = poisson_lam_h[-tail:]; poisson_lam_a = poisson_lam_a[-tail:]
    home_strength_h = home_strength_h[-tail:]; away_strength_a = away_strength_a[-tail:]

    # Market odds
    mk = up.apply(lambda r: market_probs(r.b365h, r.b365d, r.b365a, r.raw_json), axis=1, result_type="expand")
    mk.columns = ["m_h","m_d","m_a","has_m","m_over","m_src"]
    mv = up["raw_json"].apply(odds_movement).apply(pd.Series)
    mv.columns = ["mv_h","mv_d","mv_a","has_mv"]
    ou = up["raw_json"].apply(ou25_probs).apply(pd.Series)
    ou.columns = ["p_over25_mkt","ou_over","has_ou"]

    PM = mk[["m_h","m_d","m_a"]].to_numpy(float)
    PM = _repair_prob_matrix(PM, PM)

    P_elo = np.column_stack([elo_ph, elo_pd, elo_pa])

    H2 = np.array([h2h_expert_probs(n, ph, pa, gd) for n, ph, pa, gd in zip(h2n, h2ph, h2pa, h2gd)], dtype=float)
    PH2 = H2[:, :3]; H2S = H2[:, 3:4]

    # DC expert for upcoming
    dc_p = np.zeros((len(up), 3), dtype=float)
    dc_eg = np.zeros((len(up), 2), dtype=float)
    dc_o25 = np.zeros((len(up), 1), dtype=float)
    has_dc = np.zeros((len(up), 1), dtype=float)
    for i, r in enumerate(up[["competition","home_team","away_team"]].itertuples(index=False)):
        m = dc_by_comp.get(str(r.competition))
        if m is None:
            continue
        ph, p_draw, pa, eg_h, eg_a, p_over = predict_1x2(m, r.home_team, r.away_team)
        if ph > 0:
            dc_p[i] = [ph, p_draw, pa]
            dc_eg[i] = [eg_h, eg_a]
            dc_o25[i] = [p_over]
            has_dc[i] = [1.0]

    ent_elo = np.array([entropy3(p) for p in P_elo])
    entm = np.array([entropy3(p) for p in PM])
    entdc = np.array([entropy3(p) for p in dc_p])
    enth2h = np.array([entropy3(p) for p in PH2])
    dis = np.array([disagreement(np.vstack([a,b,c])) for a,b,c in zip(P_elo, PM, dc_p)])

    # Feature matrix — must match training order exactly
    X = np.column_stack([
        P_elo,
        PM, mk[["has_m","m_over","m_src"]].to_numpy(float),
        PH2, H2S,
        dc_p, has_dc, dc_eg, dc_o25,
        mv.to_numpy(float),
        ou.to_numpy(float),
        ent_elo, entm, entdc, enth2h, dis,
        rest_h, rest_a, (rest_h - rest_a),
        h2n, h2ph, h2pa, h2gd, vn, vph, vgd,
        gf_h, ga_h, pts_h, gf_a, ga_a, pts_a,
        sh_h, sot_h, cor_h, card_h,
        sh_a, sot_a, cor_a, card_a,
        cs_h, cs_a, btts_h, btts_a,
        momentum_h, momentum_a, streak_h, streak_a,
        gsup_h, gsup_a, sot_ratio_h, sot_ratio_a,
        elo_h, elo_a, (elo_h - elo_a),
        poisson_lam_h, poisson_lam_a,
        (poisson_lam_h - poisson_lam_a),
        (poisson_lam_h + poisson_lam_a),
        (poisson_lam_h * poisson_lam_a),
        home_strength_h, away_strength_a, (home_strength_h - away_strength_a),
        (gf_h - gf_a), (ga_h - ga_a), (pts_h - pts_a),
        (sh_h - sh_a), (sot_h - sot_a), (cor_h - cor_a), (card_h - card_a),
        (cs_h - cs_a), (btts_h - btts_a), (momentum_h - momentum_a), (gsup_h - gsup_a),
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    P = model.predict_proba(X)

    # No market anchor blend — model handles odds weighting internally
    n = 0
    for (mid, ph, p_draw, pa, eg_h, eg_a) in zip(up["match_id"].to_numpy(dtype=np.int64), P[:,0], P[:,1], P[:,2], dc_eg[:,0], dc_eg[:,1]):
        con.execute(
            """INSERT OR REPLACE INTO predictions
               (match_id, model_version, p_home, p_draw, p_away, eg_home, eg_away, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [int(mid), V5_MODEL_VERSION, float(ph), float(p_draw), float(pa),
             (float(eg_h) if eg_h > 0 else None),
             (float(eg_a) if eg_a > 0 else None),
             "v5 (Elo + market + H2H + DC + form — unified)"]
        )
        n += 1

    if verbose:
        print(f"[v5] predicted upcoming: {n}", flush=True)
    return n
