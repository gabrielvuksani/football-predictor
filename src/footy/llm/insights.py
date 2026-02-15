"""
Phase 3: Ollama-powered match insights and analysis

Combines GDELT news, local LLM, and cached predictions for:
- Form analysis summaries
- Team news signals 
- Match explanations
- Predictive insights
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd

from footy.cache import get_cache, cache_wrapper
from footy.db import connect
from footy.llm.ollama_client import chat
from footy.llm.news_extractor import extract_news_signal
from footy.providers.news_gdelt import fetch_team_news

log = logging.getLogger("footy.llm.insights")


# ============================================================================
# Form Analysis: Summarize recent team performance
# ============================================================================


@cache_wrapper(ttl_hours=24)
def analyze_team_form(team: str, matches_window: int = 10) -> dict:
    """
    Analyze recent form for a team using LLM.
    
    Args:
        team: Canonical team name
        matches_window: How many recent matches to consider
    
    Returns:
        {
            "team": str,
            "recent_form": str,  # "Excellent", "Good", "Mixed", "Poor"
            "key_trends": list[str],
            "momentum": float,  # -1.0 (declining) to +1.0 (improving)
            "concern_areas": list[str],
            "last_updated": str (ISO datetime)
        }
    """
    con = connect()
    
    # Get recent matches for this team
    query = f"""
    SELECT 
        utc_date, 
        home_team, 
        away_team, 
        home_goals, 
        away_goals,
        status
    FROM matches
    WHERE (home_team = '{team}' OR away_team = '{team}')
        AND status = 'FINISHED'
    ORDER BY utc_date DESC
    LIMIT {matches_window}
    """
    results = con.execute(query).df()
    
    if results.empty:
        return {
            "team": team,
            "recent_form": "No data",
            "key_trends": [],
            "momentum": 0.0,
            "concern_areas": [],
            "last_updated": datetime.now().isoformat()
        }
    
    # Calculate stats
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    
    for _, row in results.iterrows():
        if row['home_team'] == team:
            goals_for += row['home_goals']
            goals_against += row['away_goals']
            if row['home_goals'] > row['away_goals']:
                wins += 1
            elif row['home_goals'] == row['away_goals']:
                draws += 1
            else:
                losses += 1
        else:
            goals_for += row['away_goals']
            goals_against += row['home_goals']
            if row['away_goals'] > row['home_goals']:
                wins += 1
            elif row['away_goals'] == row['home_goals']:
                draws += 1
            else:
                losses += 1
    
    record = f"{wins}W-{draws}D-{losses}L"
    gd = goals_for - goals_against
    
    # Calculate form metrics from data
    def infer_form_from_record(w, d, l, gf, ga, total):
        """Generate reasonable form assessment from raw stats"""
        if total == 0:
            return "No recent matches", 0.0
        
        points = (w * 3 + d) / (total * 3)  # Proportion of max possible points
        gd_per_match = gd / total if total > 0 else 0
        
        if points >= 0.8 and gd_per_match >= 0.5:
            return "Excellent", 0.8
        elif points >= 0.6 and gd_per_match >= 0.2:
            return "Good", 0.5
        elif points >= 0.4:
            return "Mixed", 0.0
        else:
            return "Poor", -0.5
    
    form_status, momentum = infer_form_from_record(wins, draws, losses, goals_for, goals_against, matches_window)
    
    # Infer trends and concerns from data
    trends = []
    concerns = []
    if goals_for < 1.0 * matches_window / 3:
        concerns.append("Low scoring rate")
    if goals_against > 1.5 * matches_window / 3:
        concerns.append("Defensive vulnerability")
    if wins >= losses * 2:
        trends.append("Strong run")
    elif losses >= wins * 2:
        trends.append("Struggling")
    else:
        trends.append("Inconsistent")
    
    if gd > 0:
        trends.append(f"Positive goal differential (+{gd})")
    elif gd < 0:
        trends.append(f"Negative goal differential ({gd})")
    
    # Return form analysis
    result = {
        "team": team,
        "recent_form": form_status,
        "key_trends": trends[:2],  # Top 2 trends
        "momentum": momentum,
        "concern_areas": concerns[:2],  # Top 2 concerns
        "record": record,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "last_updated": datetime.now().isoformat()
    }
    
    # Try to enhance with Ollama if available
    prompt = {
        "role": "user",
        "content": f"""
Quick assessment of {team} form:
Record: {record} (last {matches_window} matches)
GF: {goals_for}, GA: {goals_against}, GD: {gd}

Return ONLY valid JSON:
{{"recent_form": "Excellent|Good|Mixed|Poor", "key_trends": ["trend1", "trend2"], "momentum": float, "concern_areas": ["area1", "area2"]}}
"""
    }
    
    try:
        response = chat([prompt])
        if response and response.strip():
            data = json.loads(response)
            result["recent_form"] = data.get("recent_form", form_status)
            result["key_trends"] = data.get("key_trends", trends)[:2]
            result["momentum"] = float(data.get("momentum", momentum))
            result["concern_areas"] = data.get("concern_areas", concerns)[:2]
    except (json.JSONDecodeError, ValueError, httpx.ConnectError):
        # Ollama unavailable or invalid response - use fallback data above
        pass
    except Exception as e:
        log.debug(f"Ollama form analysis optional enhancement failed: {e}")
    
    return result


# ============================================================================
# News Analysis: Extract team news signals
# ============================================================================


@cache_wrapper(ttl_hours=6)
def extract_team_news_signal(team: str, days_back: int = 2) -> dict:
    """
    Fetch team news from GDELT and extract signal using Ollama.
    
    Args:
        team: Canonical team name
        days_back: How many days of news to fetch
    
    Returns:
        {
            "team": str,
            "availability_score": float (-1..1),  # -1=bad news, +1=good news
            "likely_absences": list[str],  # Injured/suspended players
            "key_notes": list[str],
            "headline_count": int,
            "last_updated": str (ISO datetime)
        }
    """
    try:
        # Fetch news from GDELT
        news_df = fetch_team_news(team, days_back=days_back, max_records=20)
        
        if news_df.empty:
            log.info(f"No news found for {team}")
            return {
                "team": team,
                "availability_score": 0.0,
                "likely_absences": [],
                "key_notes": ["No recent news found"],
                "headline_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        # Convert to list of dicts for LLM
        headlines = news_df[["title", "domain", "seendate"]].to_dict("records")
        
        # Extract signal using Ollama
        signal = extract_news_signal(team, headlines)
        
        return {
            "team": team,
            "availability_score": signal.availability_score,
            "likely_absences": signal.likely_absences,
            "key_notes": signal.key_notes,
            "summary": signal.short_summary,
            "headline_count": len(headlines),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        log.warning(f"News extraction failed for {team}: {e}")
        return {
            "team": team,
            "availability_score": 0.0,
            "likely_absences": [],
            "key_notes": [f"Error: {str(e)[:50]}"],
            "headline_count": 0,
            "last_updated": datetime.now().isoformat()
        }


# ============================================================================
# Match Explanation: Generate LLM-powered insights for a specific match
# ============================================================================


@cache_wrapper(ttl_hours=24)
def explain_match(
    match_id: int, 
    home_team: str, 
    away_team: str,
    home_pred: float,
    draw_pred: float,
    away_pred: float,
    model_version: str = "v5_ultimate"
) -> dict:
    """
    Generate detailed LLM explanation for why a match has given probabilities.
    
    Args:
        match_id: Database match ID
        home_team: Home team name
        away_team: Away team name
        home_pred, draw_pred, away_pred: Model probabilities
        model_version: Which model is predicting
    
    Returns:
        {
            "match_id": int,
            "explanation": str,  # Main narrative explanation
            "key_factors": list[str],  # Top 3-5 factors influencing prediction
            "confidence_level": str,  # "Very High", "High", "Medium", "Low"
            "last_updated": str
        }
    """
    con = connect()
    
    # Get match details
    try:
        match = con.execute(f"""
            SELECT m.*
            FROM matches m
            WHERE m.match_id = {match_id}
        """).df()
        
        if match.empty:
            return {
                "match_id": match_id,
                "explanation": "Match not found",
                "key_factors": [],
                "confidence_level": "Low",
                "last_updated": datetime.now().isoformat()
            }
        
        match_row = match.iloc[0]
        
    except Exception as e:
        log.warning(f"Match lookup failed: {e}")
        match_row = {}
    
    # Get form analyses for both teams
    home_form = analyze_team_form(home_team)
    away_form = analyze_team_form(away_team)
    
    # Get news signals
    home_news = extract_team_news_signal(home_team)
    away_news = extract_team_news_signal(away_team)
    
    # Get H2H if available
    h2h_text = ""
    try:
        h2h = con.execute(f"""
            SELECT games, home_wins, draws, away_wins
            FROM h2h_stats
            WHERE team_1 = '{home_team}' AND team_2 = '{away_team}'
        """).df()
        if not h2h.empty:
            row = h2h.iloc[0]
            h2h_text = f"H2H: {row['home_wins']}W-{row['draws']}D-{row['away_wins']}W ({row['games']} games)"
    except:
        pass
    
    # Build context for LLM
    context = f"""
Match: {home_team} vs {away_team}
Model: {model_version}
Predicted: {home_pred:.1%} {draw_pred:.1%} {away_pred:.1%}

Home Team ({home_team}):
- Recent Form: {home_form.get('recent_form', 'N/A')} ({home_form.get('record', 'N/A')})
- Momentum: {home_form.get('momentum', 0):.1f}
- News: {home_news.get('summary', 'No news')}
- Availability: {home_news.get('availability_score', 0):.1f}

Away Team ({away_team}):
- Recent Form: {away_form.get('recent_form', 'N/A')} ({away_form.get('record', 'N/A')})
- Momentum: {away_form.get('momentum', 0):.1f}
- News: {away_news.get('summary', 'No news')}
- Availability: {away_news.get('availability_score', 0):.1f}

{h2h_text}

Generate a brief match explanation highlighting why the model favors these probabilities.
Focus on: form, momentum, news impact, head-to-head history.
Return JSON:
{{
    "explanation": "2-3 sentence summary",
    "key_factors": ["factor1", "factor2", "factor3"],
    "confidence_level": "Very High|High|Medium|Low"
}}
"""
    
    try:
        response = chat([{"role": "user", "content": context}])
        data = json.loads(response)
        return {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_prob": f"{home_pred:.1%}",
            "draw_prob": f"{draw_pred:.1%}",
            "away_prob": f"{away_pred:.1%}",
            "explanation": data.get("explanation", "Unable to generate"),
            "key_factors": data.get("key_factors", []),
            "confidence_level": data.get("confidence_level", "Medium"),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        log.warning(f"Match explanation failed: {e}")
        return {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "explanation": f"Error generating explanation: {str(e)[:50]}",
            "key_factors": [],
            "confidence_level": "Low",
            "last_updated": datetime.now().isoformat()
        }


# ============================================================================
# Batch Operations: Process multiple matches efficiently
# ============================================================================


def explain_matches_batch(
    matches: list[dict],
    model_version: str = "v5_ultimate"
) -> list[dict]:
    """
    Generate explanations for multiple matches.
    
    Args:
        matches: [{"match_id": int, "home_team": str, "away_team": str,
                   "home_prob": float, "draw_prob": float, "away_prob": float}, ...]
        model_version: Model being explained
    
    Returns:
        List of explanation dicts
    """
    results = []
    for match in matches:
        try:
            explanation = explain_match(
                match_id=match["match_id"],
                home_team=match["home_team"],
                away_team=match["away_team"],
                home_pred=match["home_prob"],
                draw_pred=match["draw_prob"],
                away_pred=match["away_prob"],
                model_version=model_version
            )
            results.append(explanation)
        except Exception as e:
            log.error(f"Batch explanation failed for match {match.get('match_id')}: {e}")
            results.append({
                "match_id": match.get("match_id"),
                "explanation": f"Error: {e}",
                "key_factors": [],
                "confidence_level": "Low"
            })
    
    return results


# ============================================================================
# Status & Monitoring
# ============================================================================


def get_insights_status() -> dict:
    """Get current status of insights system (cache info, LLM health)."""
    try:
        cache = get_cache()
        cache_stats = cache.get_stats()
        
        # Try an LLM health check
        try:
            response = chat([{"role": "user", "content": "Respond with 'OK'"}])
            llm_health = "OK" if response.strip() == "OK" else "Degraded"
        except Exception as e:
            llm_health = f"Error: {str(e)[:30]}"
        
        return {
            "status": "Ready",
            "llm_health": llm_health,
            "cache_predictions": cache_stats.get("total_predictions", 0),
            "cache_metadata": cache_stats.get("total_metadata", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "Error",
            "error": str(e)[:100],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Pre-match Preview: Rich AI-generated match preview
# ============================================================================


@cache_wrapper(ttl_hours=12)
def preview_match(match_id: int) -> dict:
    """
    Generate a comprehensive AI pre-match preview combining model predictions,
    form, H2H, news, and market odds.

    Returns:
        {
            "match_id": int,
            "home_team": str,
            "away_team": str,
            "preview": str,          # 3-5 sentence narrative
            "prediction": str,        # "Home Win" / "Draw" / "Away Win"
            "confidence": str,        # "High" / "Medium" / "Low"
            "value_bets": list[str],  # e.g. ["Over 2.5 @ 1.95 (model: 58%)"]
            "key_stats": list[str],   # 3-5 bullet points
        }
    """
    con = connect()

    # ---- match row ----
    row = con.execute("""
        SELECT match_id, home_team, away_team, competition, utc_date
        FROM matches WHERE match_id = ?
    """, [match_id]).fetchone()
    if not row:
        return {"match_id": match_id, "error": "match not found"}
    _, home, away, comp, utc = row

    # ---- v5 predictions ----
    pred = con.execute("""
        SELECT p_home, p_draw, p_away
        FROM predictions
        WHERE match_id = ? AND model_version LIKE 'v5%'
        ORDER BY created_at DESC LIMIT 1
    """, [match_id]).fetchone()
    ph, pd_, pa = (pred if pred else (0.33, 0.34, 0.33))

    # ---- odds ----
    odds_row = con.execute("""
        SELECT b365h, b365d, b365a
        FROM match_extras WHERE match_id = ?
    """, [match_id]).fetchone()
    odds = dict(zip(["h", "d", "a"], odds_row)) if odds_row else {}

    # ---- Elo ----
    elo = {}
    for t, label in [(home, "home"), (away, "away")]:
        e = con.execute("SELECT rating FROM elo_state WHERE team = ?", [t]).fetchone()
        if e:
            elo[label] = round(e[0], 1)

    # ---- form ----
    home_form = analyze_team_form(home)
    away_form = analyze_team_form(away)

    # ---- H2H ----
    h2h_text = ""
    try:
        h2h = con.execute("""
            SELECT games, home_wins, draws, away_wins
            FROM h2h_stats WHERE team_1 = ? AND team_2 = ?
        """, [home, away]).fetchone()
        if h2h:
            h2h_text = f"H2H ({h2h[0]} games): {h2h[1]}W-{h2h[2]}D-{h2h[3]}L for {home}"
    except Exception:
        pass

    # ---- build prompt ----
    ctx = f"""Generate a pre-match preview for:
{home} vs {away} ({comp})

Model probabilities: Home {ph:.1%}  Draw {pd_:.1%}  Away {pa:.1%}
Elo ratings: Home {elo.get('home','?')} | Away {elo.get('away','?')}
{h2h_text}
Home form: {home_form.get('recent_form','?')} ({home_form.get('record','?')})  Momentum {home_form.get('momentum',0):.1f}
Away form: {away_form.get('recent_form','?')} ({away_form.get('record','?')})  Momentum {away_form.get('momentum',0):.1f}
Market odds — H: {odds.get('h','?')} D: {odds.get('d','?')} A: {odds.get('a','?')}

Return ONLY valid JSON:
{{"preview":"3-5 sentence narrative","prediction":"Home Win|Draw|Away Win","confidence":"High|Medium|Low","value_bets":[".."],"key_stats":["..","..",".."]}}
"""
    fallback = {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "preview": f"{home} hosts {away}. Model gives {ph:.0%}/{pd_:.0%}/{pa:.0%}.",
        "prediction": ["Home Win", "Draw", "Away Win"][[ph, pd_, pa].index(max(ph, pd_, pa))],
        "confidence": "High" if max(ph, pd_, pa) > 0.55 else ("Medium" if max(ph, pd_, pa) > 0.40 else "Low"),
        "value_bets": [],
        "key_stats": [
            f"Home form: {home_form.get('record','?')}",
            f"Away form: {away_form.get('record','?')}",
            h2h_text or "No H2H data",
        ],
    }

    try:
        resp = chat([{"role": "user", "content": ctx}])
        data = json.loads(resp)
        return {**fallback, **data}
    except Exception:
        return fallback


# ============================================================================
# Value Bet Analysis: compare model vs market
# ============================================================================


def value_bet_scan(min_edge: float = 0.05) -> list[dict]:
    """
    Scan all upcoming matches for value bets where model probability
    exceeds implied odds probability by at least `min_edge`.

    Returns list of value opportunities sorted by edge descending.
    """
    con = connect()

    rows = con.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.utc_date,
               p.p_home, p.p_draw, p.p_away,
               e.b365h, e.b365d, e.b365a
        FROM matches m
        JOIN predictions p ON p.match_id = m.match_id
        LEFT JOIN match_extras e ON e.match_id = m.match_id
        WHERE m.status IN ('TIMED','SCHEDULED')
          AND p.model_version LIKE 'v5%'
        ORDER BY m.utc_date
    """).fetchall()

    values = []
    for mid, ht, at, dt, ph, pd_, pa, oh, od, oa in rows:
        for label, model_p, mkt_odds in [
            ("Home", ph, oh), ("Draw", pd_, od), ("Away", pa, oa)
        ]:
            if not mkt_odds or mkt_odds <= 1:
                continue
            implied = 1.0 / mkt_odds
            edge = model_p - implied
            if edge >= min_edge:
                values.append({
                    "match_id": mid,
                    "home_team": ht,
                    "away_team": at,
                    "date": str(dt)[:10],
                    "bet": label,
                    "model_prob": round(model_p, 3),
                    "implied_prob": round(implied, 3),
                    "edge": round(edge, 3),
                    "odds": round(mkt_odds, 2),
                })

    values.sort(key=lambda x: x["edge"], reverse=True)
    return values


def ai_value_commentary(values: list[dict], top_n: int = 5) -> str:
    """
    Ask Ollama for a short commentary on the top value bets.
    Falls back to a plain table if Ollama is unavailable.
    """
    top = values[:top_n]
    if not top:
        return "No value bets found above the edge threshold."

    lines = []
    for v in top:
        lines.append(
            f"- {v['home_team']} vs {v['away_team']}: "
            f"{v['bet']} @ {v['odds']:.2f}  "
            f"(model {v['model_prob']:.0%} vs implied {v['implied_prob']:.0%}, edge {v['edge']:+.1%})"
        )
    table = "\n".join(lines)

    ctx = (
        "You are a football betting analyst. Summarize these value bets in 3-4 sentences. "
        "Highlight the strongest edge and any risks.\n\n" + table
    )
    try:
        resp = chat([{"role": "user", "content": ctx}])
        if resp.strip():
            return resp.strip()
    except Exception:
        pass
    return "Top value bets:\n" + table


# ============================================================================
# League Round Summary
# ============================================================================


def league_round_summary(competition_code: str = "PL") -> dict:
    """
    Generate an AI-powered summary of all upcoming matches in a league.

    Returns:
        {
            "competition": str,
            "matches": int,
            "summary": str,         # AI narrative
            "headline_pick": str,    # match of the round
            "predictions": list[dict],
        }
    """
    con = connect()

    rows = con.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.utc_date,
               p.p_home, p.p_draw, p.p_away
        FROM matches m
        JOIN predictions p ON p.match_id = m.match_id
        WHERE m.status IN ('TIMED','SCHEDULED')
          AND m.competition = ?
          AND p.model_version LIKE 'v5%'
        ORDER BY m.utc_date
    """, [competition_code]).fetchall()

    if not rows:
        return {"competition": competition_code, "matches": 0,
                "summary": "No upcoming matches with predictions.", "predictions": []}

    preds = []
    lines = []
    for mid, ht, at, dt, ph, pd_, pa in rows:
        fav = ["Home", "Draw", "Away"][[ph, pd_, pa].index(max(ph, pd_, pa))]
        preds.append({"match_id": mid, "home": ht, "away": at,
                       "date": str(dt)[:10], "pred": fav,
                       "probs": f"{ph:.0%}/{pd_:.0%}/{pa:.0%}"})
        lines.append(f"{ht} vs {at} — {fav} ({max(ph,pd_,pa):.0%})")

    ctx = (
        f"You are a football analyst. Summarize this {competition_code} round in "
        "3-5 sentences. Pick the headline match and explain why.\n\n"
        + "\n".join(lines)
        + "\n\nReturn JSON: {\"summary\":\"...\", \"headline_pick\":\"Team A vs Team B\"}"
    )

    fallback = {
        "competition": competition_code,
        "matches": len(preds),
        "summary": f"{len(preds)} matches upcoming in {competition_code}.",
        "headline_pick": f"{preds[0]['home']} vs {preds[0]['away']}" if preds else "",
        "predictions": preds,
    }

    try:
        resp = chat([{"role": "user", "content": ctx}])
        data = json.loads(resp)
        fallback["summary"] = data.get("summary", fallback["summary"])
        fallback["headline_pick"] = data.get("headline_pick", fallback["headline_pick"])
    except Exception:
        pass
    return fallback


# ============================================================================
# Post-match Review: Evaluate prediction accuracy
# ============================================================================


def post_match_review(days_back: int = 3, competition_code: str | None = None) -> dict:
    """
    Review recently finished matches: compare predictions to actual results
    and generate an AI narrative on model performance.

    Returns:
        {
            "matches_reviewed": int,
            "correct": int,
            "accuracy": float,
            "review": str,          # AI narrative
            "misses": list[dict],   # biggest wrong predictions
        }
    """
    con = connect()
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    comp_filter = f"AND m.competition = '{competition_code}'" if competition_code else ""
    rows = con.execute(f"""
        SELECT m.match_id, m.home_team, m.away_team,
               m.home_goals, m.away_goals,
               p.p_home, p.p_draw, p.p_away
        FROM matches m
        JOIN predictions p ON p.match_id = m.match_id
        WHERE m.status = 'FINISHED'
          AND m.utc_date >= '{cutoff}'
          AND p.model_version LIKE 'v5%'
          {comp_filter}
        ORDER BY m.utc_date DESC
    """).fetchall()

    if not rows:
        return {"matches_reviewed": 0, "correct": 0, "accuracy": 0,
                "review": "No recent finished matches with predictions.", "misses": []}

    correct = 0
    misses = []
    lines = []
    for mid, ht, at, hg, ag, ph, pd_, pa in rows:
        pred_idx = [ph, pd_, pa].index(max(ph, pd_, pa))
        actual_idx = 0 if hg > ag else (1 if hg == ag else 2)
        ok = pred_idx == actual_idx
        if ok:
            correct += 1
        else:
            misses.append({
                "match": f"{ht} vs {at}",
                "score": f"{hg}-{ag}",
                "predicted": ["Home", "Draw", "Away"][pred_idx],
                "actual": ["Home", "Draw", "Away"][actual_idx],
                "confidence": round(max(ph, pd_, pa), 3),
            })
        lines.append(
            f"{'✓' if ok else '✗'} {ht} {hg}-{ag} {at} "
            f"(pred: {['H','D','A'][pred_idx]} {max(ph,pd_,pa):.0%})"
        )

    acc = correct / len(rows) if rows else 0
    ctx = (
        "You are a football prediction analyst reviewing recent performance.\n"
        f"Accuracy: {acc:.0%} ({correct}/{len(rows)})\n\n"
        + "\n".join(lines[:30])
        + "\n\nGive a 2-3 sentence verdict on model performance and what went wrong."
    )

    fallback_review = f"Model accuracy: {acc:.0%} ({correct}/{len(rows)} correct)"
    try:
        resp = chat([{"role": "user", "content": ctx}])
        if resp.strip():
            fallback_review = resp.strip()
    except Exception:
        pass

    return {
        "matches_reviewed": len(rows),
        "correct": correct,
        "accuracy": round(acc, 3),
        "review": fallback_review,
        "misses": sorted(misses, key=lambda x: x["confidence"], reverse=True)[:5],
    }
