"""ManagerExpert — Manager/coaching change impact detection.

Key signals:
- New manager bounce: teams with recent manager changes often overperform short-term
- Manager tenure stability: long-tenured managers = more predictable
- Tactical matchup proxy: some formations/styles counter others

Since we don't have explicit manager data in the DB, we detect manager changes
indirectly through sudden performance shifts:
- If a team's last 5 results are dramatically different from their previous 15,
  that signals a potential coaching change or tactical overhaul.
- We track the "performance discontinuity" per team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3, _pts


class ManagerExpert(Expert):
    """Detect manager/coaching changes via performance discontinuity."""

    name = "manager"

    RECENT_WINDOW = 5       # last N matches for "recent" performance
    HIST_WINDOW = 15        # previous N matches for "historical" baseline
    HOME_ADV = 0.15         # base home advantage in probability space

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        # Feature arrays
        mgr_discontinuity_h = np.zeros(n)
        mgr_discontinuity_a = np.zeros(n)
        mgr_bounce_h = np.zeros(n)
        mgr_bounce_a = np.zeros(n)
        mgr_decline_h = np.zeros(n)
        mgr_decline_a = np.zeros(n)
        mgr_stability_h = np.zeros(n)
        mgr_stability_a = np.zeros(n)

        # Per-team state: rolling list of points-per-game values
        results: dict[str, list[float]] = {}

        total_window = self.RECENT_WINDOW + self.HIST_WINDOW

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team

            for t in (h, a):
                if t not in results:
                    results[t] = []

            # --- Compute features from current state (before update) ---
            disc_h, bounce_h, decline_h, stability_h = self._compute_discontinuity(results[h])
            disc_a, bounce_a, decline_a, stability_a = self._compute_discontinuity(results[a])

            mgr_discontinuity_h[i] = disc_h
            mgr_discontinuity_a[i] = disc_a
            mgr_bounce_h[i] = bounce_h
            mgr_bounce_a[i] = bounce_a
            mgr_decline_h[i] = decline_h
            mgr_decline_a[i] = decline_a
            mgr_stability_h[i] = stability_h
            mgr_stability_a[i] = stability_a

            # --- Probability estimation ---
            # A team on a "bounce" is likely overperforming -> slight edge
            # A team in "decline" is likely underperforming -> slight penalty
            # Stability means more predictable outcomes
            adj_h = bounce_h * 0.08 - decline_h * 0.06 + stability_h * 0.02
            adj_a = bounce_a * 0.08 - decline_a * 0.06 + stability_a * 0.02

            p_h = 0.40 + self.HOME_ADV + adj_h - adj_a
            p_a = 0.30 - self.HOME_ADV / 2 + adj_a - adj_h
            p_d = 1.0 - p_h - p_a

            probs[i] = _norm3(max(p_h, 0.05), max(p_d, 0.05), max(p_a, 0.05))

            # Confidence: higher when we have enough history and there's a clear signal
            min_hist = min(len(results[h]), len(results[a]))
            signal = max(disc_h, disc_a)
            if min_hist >= total_window:
                conf[i] = min(0.85, 0.15 + signal * 0.3)
            elif min_hist >= self.RECENT_WINDOW:
                conf[i] = min(0.25, 0.05 + signal * 0.15)
            # else: conf stays 0 — not enough data

            # --- Update state with finished match result ---
            if _is_finished(r):
                hg, ag = int(r.home_goals), int(r.away_goals)
                h_ppg = _pts(hg, ag) / 3.0  # normalised to [0, 1]
                a_ppg = _pts(ag, hg) / 3.0

                results[h].append(h_ppg)
                results[a].append(a_ppg)

                # Keep rolling window bounded
                max_keep = total_window + 5  # small buffer
                if len(results[h]) > max_keep:
                    results[h] = results[h][-max_keep:]
                if len(results[a]) > max_keep:
                    results[a] = results[a][-max_keep:]

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                "mgr_discontinuity_h": mgr_discontinuity_h,
                "mgr_discontinuity_a": mgr_discontinuity_a,
                "mgr_bounce_h": mgr_bounce_h,
                "mgr_bounce_a": mgr_bounce_a,
                "mgr_decline_h": mgr_decline_h,
                "mgr_decline_a": mgr_decline_a,
                "mgr_stability_h": mgr_stability_h,
                "mgr_stability_a": mgr_stability_a,
            },
        )

    def _compute_discontinuity(
        self, history: list[float]
    ) -> tuple[float, float, float, float]:
        """Compute performance discontinuity from a team's result history.

        Returns (discontinuity, bounce, decline, stability).
        """
        if len(history) < self.RECENT_WINDOW + 3:
            # Not enough data — return neutral
            return 0.0, 0.0, 0.0, 1.0

        recent = history[-self.RECENT_WINDOW:]
        hist_end = len(history) - self.RECENT_WINDOW
        hist_start = max(0, hist_end - self.HIST_WINDOW)
        historical = history[hist_start:hist_end]

        if len(historical) < 3:
            return 0.0, 0.0, 0.0, 1.0

        recent_ppg = sum(recent) / len(recent)
        hist_ppg = sum(historical) / len(historical)

        # Performance discontinuity: how different is recent from historical
        discontinuity = abs(recent_ppg - hist_ppg) / max(hist_ppg, 0.5)

        # Positive discontinuity = improvement = new manager bounce
        diff = recent_ppg - hist_ppg
        bounce = max(0.0, diff / max(hist_ppg, 0.5))
        decline = max(0.0, -diff / max(hist_ppg, 0.5))

        # Stability = inverse of discontinuity (clamped to [0, 1])
        stability = max(0.0, 1.0 - discontinuity)

        return discontinuity, bounce, decline, stability
