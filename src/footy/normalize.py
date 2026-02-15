from __future__ import annotations
import re
import unicodedata
from typing import Optional

# Legacy aliases (kept for backward compatibility)
ALIASES = {
    # England
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Sheff Utd": "Sheffield United",
    "West Brom": "West Bromwich Albion",
    "Nott'm Forest": "Nottingham Forest",
    # Spain
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    # Italy
    "Inter": "Internazionale",
    "AC Milan": "Milan",
    # Germany (common variations)
    "Bayern Munich": "Bayern München",
    "M'gladbach": "Borussia Mönchengladbach",
    # France
    "Paris SG": "Paris Saint-Germain",
}

_SUFFIXES = (" FC", " F.C.", " CF", " C.F.", " AFC", " A.F.C.")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def canonical_team_name(name: Optional[str]) -> Optional[str]:
    """
    Convert any team name to canonical form.
    Uses comprehensive mapping from team_mapping module with fuzzy fallback.
    """
    if not name:
        return None
    
    # Try the new unified mapping system first
    try:
        from footy.team_mapping import get_canonical_name
        canonical = get_canonical_name(name)
        if canonical:
            return canonical
    except Exception:
        pass  # Fallback if team_mapping not available
    
    # Legacy normalization (fallback)
    s = str(name).strip()

    # alias first
    s = ALIASES.get(s, s)

    # normalize whitespace/punctuation
    s = re.sub(r"\s+", " ", s).strip()

    # remove common suffixes (but keep meaningful club words)
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()

    # normalize apostrophes/quotes
    s = s.replace("`", "'").replace("’", "'").replace("‘", "'").replace("´", "'")

    return s

def get_canonical_id(name: Optional[str]) -> Optional[str]:
    """Get canonical ID for a team (provider-agnostic identifier)."""
    if not name:
        return None
    try:
        from footy.team_mapping import get_canonical_id as _get_id
        canonical_id, _ = _get_id(name)
        return canonical_id
    except Exception:
        return None
