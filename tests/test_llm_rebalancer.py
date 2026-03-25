"""Tests for the LLM probability rebalancer."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from footy.models.llm_rebalancer import (
    MAX_ADJUSTMENT,
    MIN_CONFIDENCE_TO_REBALANCE,
    RebalanceResult,
    _parse_llm_response,
    batch_rebalance,
    rebalance_probabilities,
)


# ---------------------------------------------------------------------------
# _parse_llm_response tests
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    """Tests for JSON parsing from LLM output."""

    def test_parse_llm_response_json(self):
        """Plain JSON string is parsed correctly."""
        raw = json.dumps({
            "adj_home": 0.05,
            "adj_draw": -0.02,
            "adj_away": -0.03,
            "confidence": 0.7,
            "reasoning": "Derby match, draw more likely",
        })
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["adj_home"] == pytest.approx(0.05)
        assert result["adj_draw"] == pytest.approx(-0.02)
        assert result["adj_away"] == pytest.approx(-0.03)
        assert result["confidence"] == pytest.approx(0.7)
        assert result["reasoning"] == "Derby match, draw more likely"

    def test_parse_llm_response_markdown(self):
        """JSON embedded in a markdown code block is extracted."""
        raw = (
            "Here is my analysis:\n"
            "```json\n"
            '{"adj_home": 0.03, "adj_draw": 0.04, "adj_away": -0.07, '
            '"confidence": 0.6, "reasoning": "home fatigue"}\n'
            "```\n"
            "Hope that helps!"
        )
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["adj_home"] == pytest.approx(0.03)
        assert result["adj_draw"] == pytest.approx(0.04)
        assert result["adj_away"] == pytest.approx(-0.07)

    def test_parse_llm_response_markdown_no_lang(self):
        """JSON in a bare markdown code block (no language tag) is extracted."""
        raw = (
            "```\n"
            '{"adj_home": 0.0, "adj_draw": 0.0, "adj_away": 0.0, '
            '"confidence": 0.3, "reasoning": "no adjustment needed"}\n'
            "```"
        )
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["adj_home"] == 0.0

    def test_parse_llm_response_embedded(self):
        """JSON embedded in prose (no code block) is extracted via regex fallback."""
        raw = (
            'I think the adjustments should be '
            '{"adj_home": -0.05, "adj_draw": 0.08, "adj_away": -0.03, '
            '"confidence": 0.5, "reasoning": "relegation battle"} '
            'because of the high stakes.'
        )
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["adj_draw"] == pytest.approx(0.08)

    def test_parse_llm_response_garbage(self):
        """Completely unparsable text returns None."""
        assert _parse_llm_response("I have no idea what to say") is None
        assert _parse_llm_response("") is None
        assert _parse_llm_response("```\nnot json\n```") is None


# ---------------------------------------------------------------------------
# Adjustment clamping tests
# ---------------------------------------------------------------------------

class TestAdjustmentClamping:
    """Tests that adjustments are clamped to MAX_ADJUSTMENT."""

    def test_adjustments_clamped(self):
        """Adjustments exceeding +-10% are clamped."""
        # LLM returns extreme adjustments
        llm_response = json.dumps({
            "adj_home": 0.30,
            "adj_draw": -0.25,
            "adj_away": 0.20,
            "confidence": 0.9,
            "reasoning": "extreme scenario",
        })
        result = _run_rebalance_with_response(llm_response)

        assert result.applied is True
        # Each adjustment should be clamped to [-0.10, +0.10]
        for adj in result.adjustments:
            assert -MAX_ADJUSTMENT <= adj <= MAX_ADJUSTMENT

    def test_adjustments_within_range_unchanged(self):
        """Adjustments within the allowed range are not modified."""
        llm_response = json.dumps({
            "adj_home": 0.05,
            "adj_draw": -0.03,
            "adj_away": -0.02,
            "confidence": 0.8,
            "reasoning": "minor adjustment",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is True
        assert result.adjustments[0] == pytest.approx(0.05)
        assert result.adjustments[1] == pytest.approx(-0.03)
        assert result.adjustments[2] == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# Probability normalization tests
# ---------------------------------------------------------------------------

class TestProbabilityNormalization:
    """Tests that rebalanced probabilities always sum to 1."""

    def test_probabilities_sum_to_one(self):
        """Rebalanced probabilities must sum to 1.0."""
        llm_response = json.dumps({
            "adj_home": 0.05,
            "adj_draw": 0.03,
            "adj_away": -0.08,
            "confidence": 0.9,
            "reasoning": "home advantage strong",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is True
        assert sum(result.rebalanced) == pytest.approx(1.0)

    def test_probabilities_sum_to_one_with_large_adjustments(self):
        """Even with clamped extreme adjustments, probabilities sum to 1."""
        llm_response = json.dumps({
            "adj_home": 0.10,
            "adj_draw": 0.10,
            "adj_away": 0.10,
            "confidence": 1.0,
            "reasoning": "all up somehow",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is True
        assert sum(result.rebalanced) == pytest.approx(1.0)

    def test_probabilities_non_negative(self):
        """All rebalanced probabilities are non-negative (floor at 0.01)."""
        # Start with small ML probs and apply large negative adjustments
        llm_response = json.dumps({
            "adj_home": -0.10,
            "adj_draw": -0.10,
            "adj_away": 0.10,
            "confidence": 1.0,
            "reasoning": "away dominance",
        })
        result = _run_rebalance_with_response(
            llm_response, ml_probs=[0.05, 0.05, 0.90]
        )
        assert result.applied is True
        for p in result.rebalanced:
            assert p > 0


# ---------------------------------------------------------------------------
# Confidence threshold tests
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    """Tests for the confidence-based gating."""

    def test_low_confidence_not_applied(self):
        """Adjustments are not applied when LLM confidence is below threshold."""
        llm_response = json.dumps({
            "adj_home": 0.08,
            "adj_draw": -0.05,
            "adj_away": -0.03,
            "confidence": 0.1,  # Well below MIN_CONFIDENCE_TO_REBALANCE (0.3)
            "reasoning": "not very sure",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is False
        assert result.rebalanced == result.original
        assert result.adjustments == [0.0, 0.0, 0.0]

    def test_borderline_confidence_not_applied(self):
        """Confidence exactly at threshold minus epsilon is rejected."""
        llm_response = json.dumps({
            "adj_home": 0.05,
            "adj_draw": 0.02,
            "adj_away": -0.07,
            "confidence": 0.29,  # Just below 0.3
            "reasoning": "borderline case",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is False

    def test_at_threshold_applied(self):
        """Confidence at exactly the threshold is applied."""
        llm_response = json.dumps({
            "adj_home": 0.05,
            "adj_draw": 0.02,
            "adj_away": -0.07,
            "confidence": 0.3,  # Exactly at MIN_CONFIDENCE_TO_REBALANCE
            "reasoning": "just enough confidence",
        })
        result = _run_rebalance_with_response(llm_response)
        assert result.applied is True


# ---------------------------------------------------------------------------
# Fallback / error handling tests
# ---------------------------------------------------------------------------

class TestFallbackOnError:
    """Tests for graceful degradation when LLM is unavailable or fails."""

    def test_fallback_on_error(self):
        """When chat() raises an exception, original probabilities are returned."""
        ml_probs = [0.50, 0.25, 0.25]
        with patch("footy.llm.providers.chat", side_effect=RuntimeError("LLM down")):
            result = rebalance_probabilities(
                home_team="Arsenal",
                away_team="Chelsea",
                competition="PL",
                ml_probs=ml_probs,
            )
        assert result.applied is False
        assert result.rebalanced == ml_probs
        assert result.confidence == 0.0

    def test_fallback_on_empty_response(self):
        """When chat() returns empty string, original probabilities are returned."""
        ml_probs = [0.45, 0.30, 0.25]
        with patch("footy.llm.providers.chat", return_value=""):
            result = rebalance_probabilities(
                home_team="Real Madrid",
                away_team="Barcelona",
                competition="PD",
                ml_probs=ml_probs,
            )
        assert result.applied is False
        assert result.rebalanced == ml_probs

    def test_fallback_on_unparsable_response(self):
        """When chat() returns garbage, original probabilities are returned."""
        ml_probs = [0.40, 0.30, 0.30]
        with patch("footy.llm.providers.chat", return_value="Sorry, I can't help with that"):
            result = rebalance_probabilities(
                home_team="Juventus",
                away_team="Internazionale",
                competition="SA",
                ml_probs=ml_probs,
            )
        assert result.applied is False
        assert result.rebalanced == ml_probs
        assert "Failed to parse" in result.reasoning


# ---------------------------------------------------------------------------
# batch_rebalance tests
# ---------------------------------------------------------------------------

class TestBatchRebalance:
    """Tests for the batch rebalancing helper."""

    def test_batch_rebalance_returns_correct_count(self):
        """batch_rebalance returns one result per match."""
        matches = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "competition": "PL"},
            {"home_team": "Liverpool", "away_team": "Tottenham Hotspur", "competition": "PL"},
        ]
        ml_probs = np.array([[0.5, 0.25, 0.25], [0.4, 0.3, 0.3]])

        llm_response = json.dumps({
            "adj_home": 0.0, "adj_draw": 0.0, "adj_away": 0.0,
            "confidence": 0.5, "reasoning": "no change",
        })
        with patch("footy.llm.providers.chat", return_value=llm_response):
            results = batch_rebalance(matches, ml_probs)

        assert len(results) == 2
        assert all(isinstance(r, RebalanceResult) for r in results)


# ---------------------------------------------------------------------------
# Context formatting tests
# ---------------------------------------------------------------------------

class TestContextFormatting:
    """Tests that context is properly formatted in the prompt."""

    def test_derby_context_included(self):
        """Derby flag is passed through to the LLM prompt."""
        llm_response = json.dumps({
            "adj_home": 0.0, "adj_draw": 0.05, "adj_away": -0.05,
            "confidence": 0.6, "reasoning": "derby draw boost",
        })

        captured_messages = []

        def mock_chat(messages, **kwargs):
            captured_messages.extend(messages)
            return llm_response

        with patch("footy.llm.providers.chat", side_effect=mock_chat):
            result = rebalance_probabilities(
                home_team="Arsenal",
                away_team="Tottenham Hotspur",
                competition="PL",
                ml_probs=[0.45, 0.25, 0.30],
                context={"is_derby": True, "elo_diff": 50},
            )

        assert result.applied is True
        # Check that the user prompt mentions derby
        user_msg = captured_messages[-1]["content"]
        assert "DERBY" in user_msg
        assert "Elo difference" in user_msg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_rebalance_with_response(
    llm_response: str,
    ml_probs: list[float] | None = None,
) -> RebalanceResult:
    """Run rebalance_probabilities with a mocked LLM response."""
    if ml_probs is None:
        ml_probs = [0.45, 0.30, 0.25]
    with patch("footy.llm.providers.chat", return_value=llm_response):
        return rebalance_probabilities(
            home_team="Arsenal",
            away_team="Chelsea",
            competition="PL",
            ml_probs=ml_probs,
            context={"elo_diff": 80, "form_h": 2.1, "form_a": 1.5},
        )
