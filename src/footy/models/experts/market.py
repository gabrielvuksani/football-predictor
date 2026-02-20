"""MarketExpert — Odds intelligence, logit-space, Pinnacle sharp line."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.models.experts._base import Expert, ExpertResult, _f, _raw, _implied
from footy.models.advanced_math import logit, logit_space_delta, remove_overround, odds_entropy, odds_dispersion


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
        using_closing = np.zeros(n)  # P1-6 fix: flag when closing odds are used
        move_h = np.zeros(n); move_d = np.zeros(n); move_a = np.zeros(n)
        has_move = np.zeros(n)
        ou25 = np.zeros(n); ou_over = np.zeros(n); has_ou = np.zeros(n)
        # Opening odds features (always use opening for primary signal to avoid train/serve skew)
        open_ph = np.zeros(n); open_pd = np.zeros(n); open_pa = np.zeros(n)
        has_open = np.zeros(n)
        # v10: logit-space features, entropy, pinnacle-specific
        logit_h = np.zeros(n); logit_d = np.zeros(n); logit_a = np.zeros(n)
        mkt_entropy = np.zeros(n)
        pin_ph = np.zeros(n); pin_pd = np.zeros(n); pin_pa = np.zeros(n)
        has_pin = np.zeros(n)
        # v11: Asian Handicap + BTTS from The Odds API
        ah_line = np.zeros(n); ah_home = np.zeros(n); ah_away = np.zeros(n)
        has_ah = np.zeros(n)
        btts_yes = np.zeros(n); btts_no = np.zeros(n)
        has_btts = np.zeros(n)
        # v11: BTTS implied probability
        btts_implied = np.zeros(n)
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

            # P1-6 fix: Always compute opening odds baseline (available at both
            # train and serve time) to avoid closing-odds skew.
            imp_open = _implied(b365h, b365d, b365a)
            if imp_open[0] > 0:
                open_ph[i], open_pd[i], open_pa[i] = imp_open[0], imp_open[1], imp_open[2]
                has_open[i] = 1.0

            # best available odds (closing > Pinnacle > avg > max > primary)
            # The model learns via `using_closing` feature whether closing odds
            # were used, so it can discount the extra info at serve time.
            for trio, sq, is_closing in [
                ((b365ch, b365cd, b365ca), 4.0, True),
                ((psh_v, psd_v, psa_v), 3.0, False),
                ((avgh_v, avgd_v, avga_v), 2.0, False),
                ((maxh_v, maxd_v, maxa_v), 1.0, False),
                ((b365h, b365d, b365a), 0.0, False),
            ]:
                imp = _implied(trio[0], trio[1], trio[2])
                if imp[0] > 0:
                    ph[i], pd_[i], pa[i] = imp[0], imp[1], imp[2]
                    overround[i] = imp[3]
                    src_quality[i] = sq
                    has_odds[i] = 1.0
                    using_closing[i] = 1.0 if is_closing else 0.0
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

            # v10: logit-space features
            if has_odds[i]:
                logit_h[i] = logit(max(0.01, ph[i]))
                logit_d[i] = logit(max(0.01, pd_[i]))
                logit_a[i] = logit(max(0.01, pa[i]))
                mkt_entropy[i] = odds_entropy([ph[i], pd_[i], pa[i]])

            # v10: Pinnacle odds (sharpest bookmaker, least overround)
            imp_pin = _implied(psh_v, psd_v, psa_v)
            if imp_pin[0] > 0:
                pin_ph[i], pin_pd[i], pin_pa[i] = imp_pin[0], imp_pin[1], imp_pin[2]
                has_pin[i] = 1.0

            # v11: Asian Handicap from The Odds API
            _ah_l = _f(getattr(r, "odds_ah_line", None))
            _ah_h = _f(getattr(r, "odds_ah_home", None))
            _ah_a = _f(getattr(r, "odds_ah_away", None))
            if _ah_h > 1 and _ah_a > 1:
                ah_line[i] = _ah_l
                ah_home[i] = _ah_h
                ah_away[i] = _ah_a
                has_ah[i] = 1.0

            # v11: BTTS from The Odds API
            _btts_y = _f(getattr(r, "odds_btts_yes", None))
            _btts_n = _f(getattr(r, "odds_btts_no", None))
            if _btts_y > 1 and _btts_n > 1:
                btts_yes[i] = _btts_y
                btts_no[i] = _btts_n
                has_btts[i] = 1.0
                # implied BTTS probability (remove vig)
                iy = 1.0 / _btts_y
                in_ = 1.0 / _btts_n
                btts_implied[i] = iy / (iy + in_)

        return ExpertResult(
            probs=np.column_stack([ph, pd_, pa]),
            confidence=conf,
            features={
                "mkt_overround": overround, "mkt_src_quality": src_quality,
                "mkt_has_odds": has_odds,
                "mkt_using_closing": using_closing,  # P1-6: model learns closing odds bias
                "mkt_move_h": move_h, "mkt_move_d": move_d, "mkt_move_a": move_a,
                "mkt_has_move": has_move,
                "mkt_ou25": ou25, "mkt_ou_over": ou_over, "mkt_has_ou": has_ou,
                # Opening odds baseline (always available, no train/serve skew)
                "mkt_open_ph": open_ph, "mkt_open_pd": open_pd, "mkt_open_pa": open_pa,
                "mkt_has_open": has_open,
                # v10: logit-space + entropy + Pinnacle
                "mkt_logit_h": logit_h, "mkt_logit_d": logit_d, "mkt_logit_a": logit_a,
                "mkt_entropy": mkt_entropy,
                "mkt_pin_ph": pin_ph, "mkt_pin_pd": pin_pd, "mkt_pin_pa": pin_pa,
                "mkt_has_pin": has_pin,
                # v11: Asian Handicap
                "mkt_ah_line": ah_line, "mkt_ah_home": ah_home, "mkt_ah_away": ah_away,
                "mkt_has_ah": has_ah,
                # v11: BTTS odds
                "mkt_btts_yes": btts_yes, "mkt_btts_no": btts_no,
                "mkt_btts_implied": btts_implied, "mkt_has_btts": has_btts,
            },
        )
