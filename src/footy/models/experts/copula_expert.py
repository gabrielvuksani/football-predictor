"""CopulaExpert — Frank-primary copula joint distributions for 1X2 prediction.

Uses the Frank copula as the PRIMARY probability source because it correctly
models negative dependence between home/away goals (empirical Spearman rho
approx -0.143 in football). Clayton (lower-tail only, positive dependence)
and Gumbel (upper-tail only) are retained as secondary FEATURE signals —
their probabilities are exposed to the ensemble but do NOT enter the main
probability output. Disagreement between families is itself a useful signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _is_finished, _norm3
from footy.models.math.copulas import build_copula_score_matrix


class CopulaExpert(Expert):
    """
    Per-team attack/defence EMA rates -> expected goals -> copula score
    matrices (Frank, Clayton, Gumbel) -> Frank-primary 1X2 probabilities
    with Clayton/Gumbel retained as secondary feature signals.
    """

    name = "copula"

    ALPHA = 0.05          # EMA decay for attack/defence rates
    AVG = 1.35            # league-average goals per team
    MAX_GOALS = 8
    HOME_ATT_BOOST = 1.15
    AWAY_ATT_DAMP = 0.85

    # Default copula parameters
    THETA_FRANK = -1.8
    THETA_CLAYTON = 0.3
    THETA_GUMBEL = 1.15

    FAMILIES = [
        ("frank", THETA_FRANK),
        ("clayton", THETA_CLAYTON),
        ("gumbel", THETA_GUMBEL),
    ]

    def compute(self, df: pd.DataFrame) -> ExpertResult:
        n = len(df)

        # Rolling state: per-team attack (goals scored) and defence (goals conceded)
        attack: dict[str, float] = {}
        defense: dict[str, float] = {}
        game_count: dict[str, int] = {}

        # Output arrays
        probs = np.full((n, 3), 1.0 / 3.0)
        conf = np.zeros(n)

        cop_lambda_h = np.zeros(n)
        cop_lambda_a = np.zeros(n)

        # Frank-primary 1X2 (used for probs output)
        cop_frank_home = np.zeros(n)
        cop_frank_draw = np.zeros(n)
        cop_frank_away = np.zeros(n)
        cop_frank_theta = np.full(n, self.THETA_FRANK)

        # Clayton / Gumbel as feature-only signals
        cop_clayton_home = np.zeros(n)
        cop_clayton_draw = np.zeros(n)
        cop_clayton_away = np.zeros(n)
        cop_gumbel_home = np.zeros(n)
        cop_gumbel_draw = np.zeros(n)
        cop_gumbel_away = np.zeros(n)

        # Averaged 1X2 across all families (secondary feature)
        cop_avg_ph = np.zeros(n)
        cop_avg_pd = np.zeros(n)
        cop_avg_pa = np.zeros(n)

        # Market / derived features
        cop_p00 = np.zeros(n)
        cop_btts = np.zeros(n)
        cop_o25 = np.zeros(n)
        cop_entropy = np.zeros(n)
        cop_model_spread = np.zeros(n)

        def _att(t: str) -> float:
            return attack.get(t, self.AVG)

        def _def(t: str) -> float:
            return defense.get(t, self.AVG * 0.9)

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            att_h, def_h = _att(h), _def(h)
            att_a, def_a = _att(a), _def(a)

            # Expected goals with home advantage
            l_h = max(0.25, min(5.0, att_h * def_a / self.AVG * self.HOME_ATT_BOOST))
            l_a = max(0.25, min(5.0, att_a * def_h / self.AVG * self.AWAY_ATT_DAMP))
            cop_lambda_h[i] = l_h
            cop_lambda_a[i] = l_a

            # Build score matrix for each copula family and extract 1X2
            family_probs = []   # list of (ph, pd, pa)
            family_p00 = []
            family_btts = []
            family_o25 = []
            for family_name, theta in self.FAMILIES:
                mx = build_copula_score_matrix(
                    l_h, l_a,
                    theta=theta,
                    max_goals=self.MAX_GOALS,
                    copula_family=family_name,
                )

                p_home = float(np.tril(mx, -1).sum())
                p_draw = float(np.diag(mx).sum())
                p_away = float(np.triu(mx, 1).sum())
                p_home, p_draw, p_away = _norm3(p_home, p_draw, p_away)

                family_probs.append((p_home, p_draw, p_away))

                # P(0-0)
                family_p00.append(float(mx[0, 0]))

                # P(BTTS) = 1 - P(home=0) - P(away=0) + P(0-0)
                p_h0 = float(mx[0, :].sum())
                p_a0 = float(mx[:, 0].sum())
                family_btts.append(1.0 - p_h0 - p_a0 + float(mx[0, 0]))

                # P(O2.5)
                under25 = sum(
                    float(mx[hg, ag])
                    for hg in range(self.MAX_GOALS + 1)
                    for ag in range(self.MAX_GOALS + 1)
                    if hg + ag <= 2
                )
                family_o25.append(1.0 - under25)

            # --- PRIMARY output: Frank copula only ---
            frank_ph, frank_pd, frank_pa = family_probs[0]  # Frank is FAMILIES[0]
            probs[i] = [frank_ph, frank_pd, frank_pa]

            cop_frank_home[i] = frank_ph
            cop_frank_draw[i] = frank_pd
            cop_frank_away[i] = frank_pa

            # Per-family full 1X2 (Clayton, Gumbel as feature signals)
            cop_clayton_home[i] = family_probs[1][0]
            cop_clayton_draw[i] = family_probs[1][1]
            cop_clayton_away[i] = family_probs[1][2]
            cop_gumbel_home[i] = family_probs[2][0]
            cop_gumbel_draw[i] = family_probs[2][1]
            cop_gumbel_away[i] = family_probs[2][2]

            # Averaged 1X2 across all families (secondary feature, not probs)
            avg_ph = np.mean([p[0] for p in family_probs])
            avg_pd = np.mean([p[1] for p in family_probs])
            avg_pa = np.mean([p[2] for p in family_probs])
            avg_ph, avg_pd, avg_pa = _norm3(avg_ph, avg_pd, avg_pa)
            cop_avg_ph[i] = avg_ph
            cop_avg_pd[i] = avg_pd
            cop_avg_pa[i] = avg_pa

            # Market stats: use Frank-primary values (index 0)
            cop_p00[i] = float(family_p00[0])
            cop_btts[i] = float(family_btts[0])
            cop_o25[i] = float(family_o25[0])

            # Entropy of Frank-primary prediction
            p_arr = np.array([frank_ph, frank_pd, frank_pa])
            p_arr = np.clip(p_arr, 1e-12, 1.0)
            cop_entropy[i] = float(-(p_arr * np.log(p_arr)).sum())

            # Model spread: std of home-win probabilities across families
            home_probs_arr = np.array([p[0] for p in family_probs])
            cop_model_spread[i] = float(np.std(home_probs_arr))

            # Confidence based on games played
            gc_h = game_count.get(h, 0)
            gc_a = game_count.get(a, 0)
            conf[i] = min(1.0, (gc_h + gc_a) / 20.0)

            # --- Update rolling state (only for finished matches) ---
            if not _is_finished(r):
                continue

            hg, ag = int(r.home_goals), int(r.away_goals)

            if h in attack:
                attack[h] = (1 - self.ALPHA) * att_h + self.ALPHA * hg
                defense[h] = (1 - self.ALPHA) * def_h + self.ALPHA * ag
            else:
                attack[h] = float(hg) if hg > 0 else self.AVG
                defense[h] = float(ag) if ag > 0 else self.AVG * 0.9

            if a in attack:
                attack[a] = (1 - self.ALPHA) * att_a + self.ALPHA * ag
                defense[a] = (1 - self.ALPHA) * def_a + self.ALPHA * hg
            else:
                attack[a] = float(ag) if ag > 0 else self.AVG
                defense[a] = float(hg) if hg > 0 else self.AVG * 0.9

            game_count[h] = gc_h + 1
            game_count[a] = gc_a + 1

        return ExpertResult(
            probs=probs,
            confidence=conf,
            features={
                # Frank-primary 1X2 (same as probs)
                "cop_frank_home": cop_frank_home,
                "cop_frank_draw": cop_frank_draw,
                "cop_frank_away": cop_frank_away,
                "cop_frank_theta": cop_frank_theta,
                "cop_frank_primary": np.ones(n),  # indicator: Frank-only probs
                # Clayton secondary signal
                "cop_clayton_home": cop_clayton_home,
                "cop_clayton_draw": cop_clayton_draw,
                "cop_clayton_away": cop_clayton_away,
                # Gumbel secondary signal
                "cop_gumbel_home": cop_gumbel_home,
                "cop_gumbel_draw": cop_gumbel_draw,
                "cop_gumbel_away": cop_gumbel_away,
                # Averaged 1X2 across all families (secondary)
                "cop_avg_ph": cop_avg_ph,
                "cop_avg_pd": cop_avg_pd,
                "cop_avg_pa": cop_avg_pa,
                # Expected goals & market features
                "cop_lambda_h": cop_lambda_h,
                "cop_lambda_a": cop_lambda_a,
                "cop_p00": cop_p00,
                "cop_btts": cop_btts,
                "cop_o25": cop_o25,
                "cop_entropy": cop_entropy,
                "cop_model_spread": cop_model_spread,
            },
        )
