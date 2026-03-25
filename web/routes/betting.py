"""Betting picks — bet of the day, per-match best bets, same-game parlays."""
from __future__ import annotations

from fastapi import APIRouter, Query
from web.routes import con, safe_error, parse_notes

router = APIRouter(prefix="/api/betting", tags=["betting"])


def _kelly(prob: float, odds: float) -> float:
    """Full Kelly criterion stake fraction."""
    if odds <= 1.0 or prob <= 0:
        return 0.0
    return max(0.0, (prob * odds - 1) / (odds - 1))


def _edge(model_prob: float, odds: float) -> float:
    """Value edge: model probability minus implied probability."""
    if odds <= 1.0:
        return 0.0
    return model_prob - (1.0 / odds)


def _confidence_label(prob: float, edge: float) -> str:
    if edge > 0.10 and prob > 0.55:
        return "strong"
    if edge > 0.05:
        return "moderate"
    return "lean"


@router.get("/picks")
def get_betting_picks(min_edge: float = Query(0.03, ge=0.0)):
    """Get bet of the day + best bets for all upcoming matches.

    For each match, finds value across 1X2, BTTS, and O/U 2.5 markets.
    Returns the single best value bet as 'bet of the day' and per-match recommendations.
    """
    try:
        db = con()
        rows = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   m.utc_date,
                   p.p_home, p.p_draw, p.p_away, p.notes,
                   e.b365h, e.b365d, e.b365a,
                   e.psh, e.psd, e.psa,
                   e.avgh, e.avgd, e.avga,
                   e.maxh, e.maxd, e.maxa,
                   e.b365_o25, e.b365_u25,
                   e.odds_btts_yes, e.odds_btts_no
            FROM matches m
            JOIN predictions p ON p.match_id = m.match_id
            LEFT JOIN match_extras e ON e.match_id = m.match_id
            WHERE m.status IN ('SCHEDULED', 'TIMED')
              AND m.utc_date >= CURRENT_TIMESTAMP
              AND p.p_home IS NOT NULL
            ORDER BY m.utc_date ASC
        """).fetchall()

        columns = ['match_id', 'home_team', 'away_team', 'competition', 'utc_date',
                    'p_home', 'p_draw', 'p_away', 'notes',
                    'b365h', 'b365d', 'b365a', 'psh', 'psd', 'psa',
                    'avgh', 'avgd', 'avga', 'maxh', 'maxd', 'maxa',
                    'b365_o25', 'b365_u25', 'odds_btts_yes', 'odds_btts_no']

        all_match_bets = []
        all_value_bets = []  # flat list for finding bet of the day

        for row in rows:
            r = dict(zip(columns, row))
            notes = parse_notes(r.get("notes"))
            match_id = r["match_id"]
            home = r["home_team"]
            away = r["away_team"]
            comp = r["competition"]
            date_str = str(r["utc_date"])[:16] if r["utc_date"] else ""

            # Best available odds (prefer Pinnacle > avg > B365)
            def best_odds(pin, avg, b365):
                for v in [pin, avg, b365]:
                    if v and v > 1.0:
                        return float(v)
                return None

            odds_h = best_odds(r.get("psh"), r.get("avgh"), r.get("b365h"))
            odds_d = best_odds(r.get("psd"), r.get("avgd"), r.get("b365d"))
            odds_a = best_odds(r.get("psa"), r.get("avga"), r.get("b365a"))
            odds_o25 = float(r["b365_o25"]) if r.get("b365_o25") and r["b365_o25"] > 1.0 else None
            odds_u25 = float(r["b365_u25"]) if r.get("b365_u25") and r["b365_u25"] > 1.0 else None
            odds_btts_y = float(r["odds_btts_yes"]) if r.get("odds_btts_yes") and r["odds_btts_yes"] > 1.0 else None
            odds_btts_n = float(r["odds_btts_no"]) if r.get("odds_btts_no") and r["odds_btts_no"] > 1.0 else None

            p_home = float(r["p_home"]) if r.get("p_home") else 0.33
            p_draw = float(r["p_draw"]) if r.get("p_draw") else 0.33
            p_away = float(r["p_away"]) if r.get("p_away") else 0.33
            p_btts = float(notes.get("btts", 0.5)) if notes else 0.5
            p_o25 = float(notes.get("o25", 0.5)) if notes else 0.5

            match_picks = []

            # 1X2 markets
            for label, prob, odds in [("Home", p_home, odds_h), ("Draw", p_draw, odds_d), ("Away", p_away, odds_a)]:
                if odds and odds > 1.0:
                    e = _edge(prob, odds)
                    if e >= min_edge:
                        pick = {
                            "market": "1X2",
                            "selection": label,
                            "odds": round(odds, 2),
                            "model_prob": round(prob, 3),
                            "implied_prob": round(1.0 / odds, 3),
                            "edge": round(e, 3),
                            "kelly": round(_kelly(prob, odds) * 100, 1),
                            "confidence": _confidence_label(prob, e),
                        }
                        match_picks.append(pick)

            # BTTS market
            if odds_btts_y and odds_btts_y > 1.0:
                e = _edge(p_btts, odds_btts_y)
                if e >= min_edge:
                    match_picks.append({
                        "market": "BTTS", "selection": "Yes",
                        "odds": round(odds_btts_y, 2),
                        "model_prob": round(p_btts, 3),
                        "implied_prob": round(1.0 / odds_btts_y, 3),
                        "edge": round(e, 3),
                        "kelly": round(_kelly(p_btts, odds_btts_y) * 100, 1),
                        "confidence": _confidence_label(p_btts, e),
                    })
            if odds_btts_n and odds_btts_n > 1.0:
                e = _edge(1 - p_btts, odds_btts_n)
                if e >= min_edge:
                    match_picks.append({
                        "market": "BTTS", "selection": "No",
                        "odds": round(odds_btts_n, 2),
                        "model_prob": round(1 - p_btts, 3),
                        "implied_prob": round(1.0 / odds_btts_n, 3),
                        "edge": round(e, 3),
                        "kelly": round(_kelly(1 - p_btts, odds_btts_n) * 100, 1),
                        "confidence": _confidence_label(1 - p_btts, e),
                    })

            # Over/Under 2.5
            if odds_o25 and odds_o25 > 1.0:
                e = _edge(p_o25, odds_o25)
                if e >= min_edge:
                    match_picks.append({
                        "market": "O/U 2.5", "selection": "Over",
                        "odds": round(odds_o25, 2),
                        "model_prob": round(p_o25, 3),
                        "implied_prob": round(1.0 / odds_o25, 3),
                        "edge": round(e, 3),
                        "kelly": round(_kelly(p_o25, odds_o25) * 100, 1),
                        "confidence": _confidence_label(p_o25, e),
                    })
            if odds_u25 and odds_u25 > 1.0:
                e = _edge(1 - p_o25, odds_u25)
                if e >= min_edge:
                    match_picks.append({
                        "market": "O/U 2.5", "selection": "Under",
                        "odds": round(odds_u25, 2),
                        "model_prob": round(1 - p_o25, 3),
                        "implied_prob": round(1.0 / odds_u25, 3),
                        "edge": round(e, 3),
                        "kelly": round(_kelly(1 - p_o25, odds_u25) * 100, 1),
                        "confidence": _confidence_label(1 - p_o25, e),
                    })

            # Sort by edge descending
            match_picks.sort(key=lambda x: x["edge"], reverse=True)
            best_bet = match_picks[0] if match_picks else None

            # Build same-game parlay from independent markets (1X2 + BTTS + O/U)
            parlay = None
            if len(match_picks) >= 2:
                # Pick best from each market category
                by_market = {}
                for p in match_picks:
                    mk = p["market"]
                    if mk not in by_market or p["edge"] > by_market[mk]["edge"]:
                        by_market[mk] = p
                if len(by_market) >= 2:
                    legs = list(by_market.values())[:3]
                    combo_odds = 1.0
                    combo_prob = 1.0
                    for leg in legs:
                        combo_odds *= leg["odds"]
                        combo_prob *= leg["model_prob"]
                    parlay = {
                        "legs": legs,
                        "combined_odds": round(combo_odds, 2),
                        "combined_prob": round(combo_prob, 3),
                        "combined_edge": round(combo_prob - (1.0 / combo_odds), 3) if combo_odds > 1 else 0,
                    }

            match_entry = {
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "competition": comp,
                "date": date_str,
                "best_bet": best_bet,
                "all_bets": match_picks[:5],  # top 5
                "parlay": parlay,
                "p_home": round(p_home, 3),
                "p_draw": round(p_draw, 3),
                "p_away": round(p_away, 3),
            }
            all_match_bets.append(match_entry)

            # Track for bet of the day
            for pick in match_picks:
                all_value_bets.append({**pick, "match_id": match_id, "home_team": home, "away_team": away, "competition": comp, "date": date_str})

        # Bet of the day: highest edge with strong confidence
        all_value_bets.sort(key=lambda x: (x["confidence"] == "strong", x["edge"]), reverse=True)
        bot = None
        if all_value_bets:
            top = all_value_bets[0]
            bot = {
                "match_id": top["match_id"],
                "home_team": top["home_team"],
                "away_team": top["away_team"],
                "competition": top["competition"],
                "date": top["date"],
                "pick": {
                    "market": top["market"],
                    "selection": top["selection"],
                    "odds": top["odds"],
                    "model_prob": top["model_prob"],
                    "edge": top["edge"],
                    "kelly": top["kelly"],
                    "confidence": top["confidence"],
                },
            }

        return {
            "bet_of_the_day": bot,
            "matches": all_match_bets,
            "total_value_bets": len(all_value_bets),
        }
    except Exception as exc:
        return safe_error(exc, "betting picks")


@router.get("/match/{match_id}")
def get_match_bets(match_id: int):
    """Get all betting recommendations for a specific match."""
    try:
        db = con()
        row = db.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.competition,
                   m.utc_date,
                   p.p_home, p.p_draw, p.p_away, p.notes,
                   e.b365h, e.b365d, e.b365a,
                   e.psh, e.psd, e.psa,
                   e.avgh, e.avgd, e.avga,
                   e.maxh, e.maxd, e.maxa,
                   e.b365_o25, e.b365_u25,
                   e.odds_btts_yes, e.odds_btts_no,
                   e.odds_ah_line, e.odds_ah_home, e.odds_ah_away
            FROM matches m
            JOIN predictions p ON p.match_id = m.match_id
            LEFT JOIN match_extras e ON e.match_id = m.match_id
            WHERE m.match_id = ?
        """, [match_id]).fetchone()

        if not row:
            return {"error": "Match not found or no prediction available"}

        columns = ['match_id', 'home_team', 'away_team', 'competition', 'utc_date',
                    'p_home', 'p_draw', 'p_away', 'notes',
                    'b365h', 'b365d', 'b365a', 'psh', 'psd', 'psa',
                    'avgh', 'avgd', 'avga', 'maxh', 'maxd', 'maxa',
                    'b365_o25', 'b365_u25', 'odds_btts_yes', 'odds_btts_no',
                    'odds_ah_line', 'odds_ah_home', 'odds_ah_away']
        r = dict(zip(columns, row))
        notes = parse_notes(r.get("notes"))

        p_home = float(r["p_home"]) if r.get("p_home") else 0.33
        p_draw = float(r["p_draw"]) if r.get("p_draw") else 0.33
        p_away = float(r["p_away"]) if r.get("p_away") else 0.33
        p_btts = float(notes.get("btts", 0.5)) if notes else 0.5
        p_o25 = float(notes.get("o25", 0.5)) if notes else 0.5

        def best_odds(pin, avg, b365):
            for v in [pin, avg, b365]:
                if v and v > 1.0:
                    return float(v)
            return None

        markets = []

        # 1X2
        for label, prob, odds_val in [
            ("Home", p_home, best_odds(r.get("psh"), r.get("avgh"), r.get("b365h"))),
            ("Draw", p_draw, best_odds(r.get("psd"), r.get("avgd"), r.get("b365d"))),
            ("Away", p_away, best_odds(r.get("psa"), r.get("avga"), r.get("b365a"))),
        ]:
            if odds_val and odds_val > 1.0:
                markets.append({
                    "market": "1X2", "selection": label,
                    "odds": round(odds_val, 2),
                    "model_prob": round(prob, 3),
                    "implied_prob": round(1.0 / odds_val, 3),
                    "edge": round(_edge(prob, odds_val), 3),
                    "kelly": round(_kelly(prob, odds_val) * 100, 1),
                    "confidence": _confidence_label(prob, _edge(prob, odds_val)),
                    "is_value": _edge(prob, odds_val) > 0.03,
                })

        # BTTS
        for label, prob, odds_val in [("Yes", p_btts, r.get("odds_btts_yes")), ("No", 1 - p_btts, r.get("odds_btts_no"))]:
            if odds_val and float(odds_val) > 1.0:
                odds_val = float(odds_val)
                markets.append({
                    "market": "BTTS", "selection": label,
                    "odds": round(odds_val, 2),
                    "model_prob": round(prob, 3),
                    "implied_prob": round(1.0 / odds_val, 3),
                    "edge": round(_edge(prob, odds_val), 3),
                    "kelly": round(_kelly(prob, odds_val) * 100, 1),
                    "confidence": _confidence_label(prob, _edge(prob, odds_val)),
                    "is_value": _edge(prob, odds_val) > 0.03,
                })

        # O/U 2.5
        for label, prob, odds_val in [("Over", p_o25, r.get("b365_o25")), ("Under", 1 - p_o25, r.get("b365_u25"))]:
            if odds_val and float(odds_val) > 1.0:
                odds_val = float(odds_val)
                markets.append({
                    "market": "O/U 2.5", "selection": label,
                    "odds": round(odds_val, 2),
                    "model_prob": round(prob, 3),
                    "implied_prob": round(1.0 / odds_val, 3),
                    "edge": round(_edge(prob, odds_val), 3),
                    "kelly": round(_kelly(prob, odds_val) * 100, 1),
                    "confidence": _confidence_label(prob, _edge(prob, odds_val)),
                    "is_value": _edge(prob, odds_val) > 0.03,
                })

        # Asian Handicap
        ah_line = r.get("odds_ah_line")
        ah_home = r.get("odds_ah_home")
        _ah_away = r.get("odds_ah_away")
        if ah_line and ah_home and float(ah_home) > 1.0:
            markets.append({
                "market": "Asian HC", "selection": f"Home {float(ah_line):+.1f}",
                "odds": round(float(ah_home), 2),
                "model_prob": None,
                "implied_prob": round(1.0 / float(ah_home), 3),
                "edge": None,
                "kelly": None,
                "confidence": "info",
                "is_value": False,
            })

        markets.sort(key=lambda x: x.get("edge") or -999, reverse=True)
        best = next((m for m in markets if m.get("is_value")), markets[0] if markets else None)

        # Same-game parlay
        parlay = None
        value_by_market = {}
        for m in markets:
            mk = m["market"]
            if m.get("is_value") and mk not in value_by_market:
                value_by_market[mk] = m
        if len(value_by_market) >= 2:
            legs = list(value_by_market.values())[:3]
            combo_odds = 1.0
            combo_prob = 1.0
            for leg in legs:
                combo_odds *= leg["odds"]
                combo_prob *= leg["model_prob"]
            parlay = {
                "legs": legs,
                "combined_odds": round(combo_odds, 2),
                "combined_prob": round(combo_prob, 3),
            }

        return {
            "match_id": match_id,
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "competition": r["competition"],
            "best_bet": best,
            "markets": markets,
            "parlay": parlay,
        }
    except Exception as exc:
        return safe_error(exc, "match bets")
