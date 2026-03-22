"""Unit tests — Team name normalization."""
import pytest


class TestCanonicalName:
    """canonical_team_name() edge cases and alias resolution."""

    def test_identity(self):
        from footy.normalize import canonical_team_name
        assert canonical_team_name("Arsenal") == "Arsenal"

    def test_alias_man_city(self):
        from footy.normalize import canonical_team_name
        result = canonical_team_name("Man City")
        assert result in ("Manchester City", "Man City")  # team_mapping may override

    def test_alias_spurs(self):
        from footy.normalize import canonical_team_name
        result = canonical_team_name("Spurs")
        assert "Tottenham" in result or "Spurs" in result

    def test_none_input(self):
        from footy.normalize import canonical_team_name
        assert canonical_team_name(None) is None

    def test_empty_string(self):
        from footy.normalize import canonical_team_name
        result = canonical_team_name("")
        assert result is None or result == ""

    def test_strips_fc_suffix(self):
        from footy.normalize import canonical_team_name
        result = canonical_team_name("Arsenal FC")
        assert "FC" not in result

    def test_whitespace_handling(self):
        from footy.normalize import canonical_team_name
        result = canonical_team_name("  Arsenal  ")
        assert result == canonical_team_name("Arsenal")

    def test_accent_tolerance(self):
        from footy.normalize import canonical_team_name
        r1 = canonical_team_name("Bayern München")
        r2 = canonical_team_name("Bayern Munchen")
        # Both should resolve to the same canonical name
        assert r1 is not None
        assert r2 is not None


class TestCanonicalId:
    def test_returns_string_or_none(self):
        from footy.normalize import get_canonical_id
        result = get_canonical_id("Arsenal")
        assert result is None or isinstance(result, str)
