"""LLM Probability Rebalancer — AI-assisted probability adjustment.

Inspired by the WINNER12 W-5 framework which uses an LLM as a
"Probability Rebalancer" achieving +10% on draws and +25% on upsets.

The ML model produces baseline probabilities. The LLM analyzes qualitative
context (news, injuries, motivation, derby intensity) that the ML can't
capture, and suggests small probability adjustments.

Key design:
- Adjustments are SMALL (max ±10% per outcome)
- Probabilities always re-normalized to sum to 1.0
- Falls back to ML probabilities if LLM fails
- Only activates for matches where context is available
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

MAX_ADJUSTMENT = 0.10  # Maximum ±10% shift per outcome
MIN_CONFIDENCE_TO_REBALANCE = 0.3  # Only rebalance if LLM is somewhat confident


@dataclass
class RebalanceResult:
    """Result of LLM probability rebalancing."""
    original: list[float]     # [p_home, p_draw, p_away] from ML
    rebalanced: list[float]   # [p_home, p_draw, p_away] after LLM adjustment
    adjustments: list[float]  # [adj_home, adj_draw, adj_away]
    reasoning: str            # LLM's explanation
    confidence: float         # LLM's self-reported confidence
    applied: bool             # Whether rebalancing was actually applied


def rebalance_probabilities(
    home_team: str,
    away_team: str,
    competition: str,
    ml_probs: list[float],
    context: dict | None = None,
) -> RebalanceResult:
    """Use LLM to adjust ML probabilities based on qualitative context.

    Args:
        home_team: Home team name
        away_team: Away team name
        competition: Competition code (PL, PD, SA, etc.)
        ml_probs: [p_home, p_draw, p_away] from the ML model
        context: Optional dict with additional context (form, injuries, etc.)

    Returns:
        RebalanceResult with original and adjusted probabilities.
    """
    original = list(ml_probs)

    # Build context string
    ctx_parts = []
    if context:
        if context.get("elo_diff"):
            ctx_parts.append(f"Elo difference: {context['elo_diff']:+.0f} (positive = home stronger)")
        if context.get("form_h") is not None:
            ctx_parts.append(f"Home recent form: {context['form_h']:.1f} PPG")
        if context.get("form_a") is not None:
            ctx_parts.append(f"Away recent form: {context['form_a']:.1f} PPG")
        if context.get("is_derby"):
            ctx_parts.append("This is a DERBY match (historically unpredictable)")
        if context.get("is_relegation"):
            ctx_parts.append("Relegation battle involved (high motivation)")
        if context.get("fatigue_diff"):
            ctx_parts.append(f"Fatigue difference: {context['fatigue_diff']:+.1f} days rest advantage to home")
        if context.get("h2h_draw_rate"):
            ctx_parts.append(f"H2H draw rate: {context['h2h_draw_rate']:.0%}")
        if context.get("league_draw_rate"):
            ctx_parts.append(f"League draw rate: {context['league_draw_rate']:.0%}")

    context_str = "\n".join(ctx_parts) if ctx_parts else "No additional context available."

    prompt = f"""You are an expert football analyst. A machine learning model has produced the following match probabilities:

Match: {home_team} vs {away_team} ({competition})
ML Probabilities: Home {ml_probs[0]:.1%} | Draw {ml_probs[1]:.1%} | Away {ml_probs[2]:.1%}

Context:
{context_str}

Based on this context, should the probabilities be adjusted? Consider:
- Are there factors the ML model likely missed (derby intensity, recent managerial changes, key player injuries, etc.)?
- Is the draw probability realistic for this type of match?
- Are there upset indicators that the ML may have underweighted?

Respond in EXACTLY this JSON format:
{{"adj_home": 0.0, "adj_draw": 0.0, "adj_away": 0.0, "confidence": 0.5, "reasoning": "brief explanation"}}

Where adj values are between -0.10 and +0.10 (adjustments to add to the ML probabilities).
If you think the ML probabilities are reasonable, use all zeros for adjustments.
Confidence is 0.0-1.0 indicating how confident you are in your adjustments."""

    try:
        from footy.llm.providers import chat
        response = chat([
            {"role": "system", "content": "You are a football prediction expert. Respond only in the exact JSON format requested."},
            {"role": "user", "content": prompt},
        ])

        if not response:
            return RebalanceResult(
                original=original, rebalanced=original,
                adjustments=[0.0, 0.0, 0.0], reasoning="LLM unavailable",
                confidence=0.0, applied=False,
            )

        # Parse JSON from response
        parsed = _parse_llm_response(response)
        if parsed is None:
            return RebalanceResult(
                original=original, rebalanced=original,
                adjustments=[0.0, 0.0, 0.0], reasoning="Failed to parse LLM response",
                confidence=0.0, applied=False,
            )

        adj_h = np.clip(parsed.get("adj_home", 0.0), -MAX_ADJUSTMENT, MAX_ADJUSTMENT)
        adj_d = np.clip(parsed.get("adj_draw", 0.0), -MAX_ADJUSTMENT, MAX_ADJUSTMENT)
        adj_a = np.clip(parsed.get("adj_away", 0.0), -MAX_ADJUSTMENT, MAX_ADJUSTMENT)
        confidence = np.clip(parsed.get("confidence", 0.0), 0.0, 1.0)
        reasoning = parsed.get("reasoning", "")

        # Only apply if LLM is confident enough
        if confidence < MIN_CONFIDENCE_TO_REBALANCE:
            return RebalanceResult(
                original=original, rebalanced=original,
                adjustments=[0.0, 0.0, 0.0],
                reasoning=f"LLM confidence too low ({confidence:.2f}): {reasoning}",
                confidence=confidence, applied=False,
            )

        # Apply adjustments with confidence weighting
        new_h = ml_probs[0] + adj_h * confidence
        new_d = ml_probs[1] + adj_d * confidence
        new_a = ml_probs[2] + adj_a * confidence

        # Ensure non-negative
        new_h = max(0.01, new_h)
        new_d = max(0.01, new_d)
        new_a = max(0.01, new_a)

        # Re-normalize to sum to 1
        total = new_h + new_d + new_a
        rebalanced = [new_h / total, new_d / total, new_a / total]

        return RebalanceResult(
            original=original,
            rebalanced=rebalanced,
            adjustments=[float(adj_h), float(adj_d), float(adj_a)],
            reasoning=reasoning,
            confidence=float(confidence),
            applied=True,
        )

    except Exception as e:
        log.debug("LLM rebalancing failed: %s", e)
        return RebalanceResult(
            original=original, rebalanced=original,
            adjustments=[0.0, 0.0, 0.0], reasoning=f"Error: {e}",
            confidence=0.0, applied=False,
        )


def _parse_llm_response(response: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try direct JSON parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the response
    json_match = re.search(r'\{[^{}]*"adj_home"[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def batch_rebalance(
    matches: list[dict],
    ml_probs_batch: np.ndarray,
) -> list[RebalanceResult]:
    """Rebalance a batch of match predictions.

    Args:
        matches: List of dicts with keys: home_team, away_team, competition, context
        ml_probs_batch: (n, 3) array of ML probabilities

    Returns:
        List of RebalanceResult, one per match.
    """
    results = []
    for i, match in enumerate(matches):
        result = rebalance_probabilities(
            home_team=match["home_team"],
            away_team=match["away_team"],
            competition=match.get("competition", ""),
            ml_probs=ml_probs_batch[i].tolist(),
            context=match.get("context"),
        )
        results.append(result)
    return results
