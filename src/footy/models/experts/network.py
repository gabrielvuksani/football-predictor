"""NetworkStrengthExpert — PageRank, Massey, Colley network-based ratings.

Models the league as a directed weighted graph. Three complementary
algorithms produce team strength ratings.

Performance fix: caches computed ratings per competition and only
recomputes when the result buffer changes (avoids O(N*M^2) per row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.network_strength import (
    pagerank_ratings,
    massey_ratings,
    colley_ratings,
)

_WINDOW = 120  # only use last N finished matches per competition


class NetworkStrengthExpert(Expert):
    name = "network"

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Output arrays
        pr_h = np.zeros(n); pr_a = np.zeros(n); pr_diff = np.zeros(n)
        ma_h = np.zeros(n); ma_a = np.zeros(n); ma_diff = np.zeros(n)
        co_h = np.zeros(n); co_a = np.zeros(n); co_diff = np.zeros(n)
        composite_diff = np.zeros(n)
        agreement = np.zeros(n)
        probs = np.full((n, 3), 1.0 / 3)
        conf = np.zeros(n)

        # Per-competition finished result buffer
        comp_results: dict[str, list[dict]] = {}
        # Cache: comp -> (buffer_len, teams_set, pr_ratings, ma_ratings, co_ratings)
        _cache: dict[str, tuple[int, dict, dict, dict]] = {}

        for i, r in enumerate(df.itertuples(index=False)):
            h = getattr(r, "home_team", "")
            a = getattr(r, "away_team", "")
            comp = str(getattr(r, "competition", "UNK"))

            buf = comp_results.get(comp, [])

            # Compute features from current buffer
            if len(buf) >= 30:
                # Only recompute if buffer has grown since last computation
                cached = _cache.get(comp)
                if cached is not None and cached[0] == len(buf):
                    # Reuse cached ratings
                    pr_ratings, ma_ratings, co_ratings = cached[1], cached[2], cached[3]
                else:
                    # Buffer changed — recompute
                    teams = sorted({d["home_team"] for d in buf} | {d["away_team"] for d in buf})
                    pr_ratings = pagerank_ratings(teams, buf)
                    ma_ratings = massey_ratings(teams, buf)
                    co_ratings = colley_ratings(teams, buf)
                    _cache[comp] = (len(buf), pr_ratings, ma_ratings, co_ratings)

                if h in pr_ratings and a in pr_ratings:
                    n_teams = len(pr_ratings)
                    pr_h[i] = pr_ratings.get(h, 1.0 / max(n_teams, 1))
                    pr_a[i] = pr_ratings.get(a, 1.0 / max(n_teams, 1))
                    pr_diff[i] = pr_h[i] - pr_a[i]

                    ma_h[i] = ma_ratings.get(h, 0.0)
                    ma_a[i] = ma_ratings.get(a, 0.0)
                    ma_diff[i] = ma_h[i] - ma_a[i]

                    co_h[i] = co_ratings.get(h, 0.5)
                    co_a[i] = co_ratings.get(a, 0.5)
                    co_diff[i] = co_h[i] - co_a[i]

                    # Composite: normalised diffs
                    diffs = [pr_diff[i] * 100, ma_diff[i], co_diff[i] * 10]
                    norm_diffs = [np.tanh(d / 2.0) for d in diffs]
                    composite_diff[i] = float(np.mean(norm_diffs))

                    # Agreement
                    signs = [np.sign(d) for d in diffs]
                    if signs[0] == signs[1] == signs[2] and signs[0] != 0:
                        agreement[i] = 1.0
                    elif signs.count(0) >= 2:
                        agreement[i] = 0.5
                    else:
                        agreement[i] = sum(1.0 for s in signs if s == np.sign(composite_diff[i])) / 3.0

                    # Probability estimate from composite strength
                    cd = composite_diff[i]
                    p_h = 0.45 + cd * 0.30
                    p_a = 0.28 - cd * 0.22
                    p_d = 1.0 - p_h - p_a
                    probs[i] = _norm3(max(0.05, p_h), max(0.12, p_d), max(0.05, p_a))

                    conf[i] = min(1.0, len(buf) / 100.0)

            # Update buffer with finished results only
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                buf.append({
                    "home_team": h,
                    "away_team": a,
                    "home_goals": hg,
                    "away_goals": ag,
                })
                # Trim to window
                if len(buf) > _WINDOW:
                    buf = buf[-_WINDOW:]
                comp_results[comp] = buf

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "net_pr_h": pr_h, "net_pr_a": pr_a, "net_pr_diff": pr_diff,
                "net_ma_h": ma_h, "net_ma_a": ma_a, "net_ma_diff": ma_diff,
                "net_co_h": co_h, "net_co_a": co_a, "net_co_diff": co_diff,
                "net_composite_diff": composite_diff,
                "net_agreement": agreement,
            },
        )
