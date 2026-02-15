"""
Unified team name mapping and normalization across all data providers.

Covers ALL 291+ teams observed in database across:
- football-data.org (full official names like "FC Internazionale Milano")
- football-data.co.uk (short names like "Inter")
- api-football / other sources

Key concepts:
- canonical_id: universal slug (e.g., "manchester-city")
- canonical_name: first entry in the list — the display name
- All other entries are known aliases/provider variants
"""

from __future__ import annotations
import re
import unicodedata
from typing import Optional, Dict, Tuple
from difflib import SequenceMatcher
import duckdb

# ============================================================================
# COMPREHENSIVE TEAM MAPPINGS (all DB teams covered)
# First entry = canonical display name
# ============================================================================

TEAM_MAPPINGS: Dict[str, list[str]] = {
    # ======================== ENGLAND ========================
    "arsenal": ["Arsenal", "Arsenal FC"],
    "aston-villa": ["Aston Villa", "Aston Villa FC"],
    "bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "brentford": ["Brentford", "Brentford FC"],
    "brighton": ["Brighton", "Brighton & Hove Albion", "Brighton & Hove Albion FC", "Brighton & HA"],
    "burnley": ["Burnley", "Burnley FC"],
    "chelsea": ["Chelsea", "Chelsea FC"],
    "crystal-palace": ["Crystal Palace", "Crystal Palace FC"],
    "everton": ["Everton", "Everton FC"],
    "fulham": ["Fulham", "Fulham FC"],
    "huddersfield": ["Huddersfield", "Huddersfield Town"],
    "ipswich-town": ["Ipswich Town", "Ipswich"],
    "leeds-united": ["Leeds United", "Leeds United FC", "Leeds"],
    "leicester-city": ["Leicester City", "Leicester"],
    "liverpool": ["Liverpool", "Liverpool FC"],
    "luton-town": ["Luton", "Luton Town"],
    "manchester-city": ["Manchester City", "Manchester City FC", "Man City"],
    "manchester-united": ["Manchester United", "Manchester United FC", "Man United", "Man Utd"],
    "newcastle-united": ["Newcastle United", "Newcastle United FC", "Newcastle"],
    "norwich-city": ["Norwich", "Norwich City"],
    "nottingham-forest": ["Nottingham Forest", "Nottingham Forest FC", "Nott'm Forest", "Notts Forest"],
    "sheffield-united": ["Sheffield United", "Sheff Utd", "Sheff United"],
    "southampton": ["Southampton", "Southampton FC"],
    "sunderland": ["Sunderland", "Sunderland AFC"],
    "tottenham-hotspur": ["Tottenham Hotspur", "Tottenham Hotspur FC", "Tottenham", "Spurs"],
    "watford": ["Watford", "Watford FC"],
    "west-bromwich-albion": ["West Bromwich Albion", "West Brom", "WBA"],
    "west-ham-united": ["West Ham", "West Ham United", "West Ham United FC"],
    "wolverhampton-wanderers": ["Wolverhampton Wanderers", "Wolverhampton Wanderers FC", "Wolves"],
    "cardiff-city": ["Cardiff", "Cardiff City"],

    # ======================== SPAIN ========================
    "atletico-madrid": ["Atletico Madrid", "Club Atlético de Madrid", "Atlético Madrid", "Ath Madrid", "Atl Madrid"],
    "athletic-bilbao": ["Athletic Bilbao", "Athletic Club", "Ath Bilbao"],
    "alaves": ["Alaves", "Deportivo Alavés", "Alavés"],
    "almeria": ["Almeria", "UD Almería"],
    "barcelona": ["Barcelona", "FC Barcelona"],
    "cadiz": ["Cadiz", "Cádiz CF"],
    "celta-vigo": ["Celta", "RC Celta de Vigo", "Celta Vigo", "Celta de Vigo"],
    "eibar": ["Eibar", "SD Eibar"],
    "elche": ["Elche", "Elche CF"],
    "espanyol": ["Espanol", "RCD Espanyol de Barcelona", "Espanyol", "RCD Espanyol"],
    "getafe": ["Getafe", "Getafe CF"],
    "girona": ["Girona", "Girona FC"],
    "granada": ["Granada", "Granada CF"],
    "huesca": ["Huesca", "SD Huesca"],
    "las-palmas": ["Las Palmas", "UD Las Palmas"],
    "leganes": ["Leganes", "CD Leganés"],
    "levante": ["Levante", "Levante UD"],
    "mallorca": ["Mallorca", "RCD Mallorca"],
    "osasuna": ["Osasuna", "CA Osasuna"],
    "rayo-vallecano": ["Vallecano", "Rayo Vallecano de Madrid", "Rayo Vallecano"],
    "real-betis": ["Real Betis", "Real Betis Balompié", "Betis"],
    "real-madrid": ["Real Madrid", "Real Madrid CF"],
    "real-oviedo": ["Oviedo", "Real Oviedo"],
    "real-sociedad": ["Sociedad", "Real Sociedad de Fútbol", "Real Sociedad"],
    "real-valladolid": ["Valladolid", "Real Valladolid"],
    "sevilla": ["Sevilla", "Sevilla FC", "Seville"],
    "valencia": ["Valencia", "Valencia CF"],
    "villarreal": ["Villarreal", "Villarreal CF"],

    # ======================== ITALY ========================
    "ac-milan": ["AC Milan", "Milan", "A.C. Milan"],
    "atalanta": ["Atalanta", "Atalanta BC"],
    "benevento": ["Benevento", "Benevento Calcio"],
    "bologna": ["Bologna", "Bologna FC 1909"],
    "brescia": ["Brescia", "Brescia Calcio"],
    "cagliari": ["Cagliari", "Cagliari Calcio"],
    "chievo": ["Chievo", "Chievo Verona"],
    "como": ["Como", "Como 1907"],
    "cremonese": ["Cremonese", "US Cremonese"],
    "crotone": ["Crotone", "FC Crotone"],
    "empoli": ["Empoli", "Empoli FC"],
    "fiorentina": ["Fiorentina", "ACF Fiorentina"],
    "frosinone": ["Frosinone", "Frosinone Calcio"],
    "genoa": ["Genoa", "Genoa CFC"],
    "hellas-verona": ["Verona", "Hellas Verona FC", "Hellas Verona"],
    "internazionale": ["Internazionale", "FC Internazionale Milano", "Inter", "Inter Milan"],
    "juventus": ["Juventus", "Juventus FC", "Juve"],
    "lazio": ["Lazio", "SS Lazio"],
    "lecce": ["Lecce", "US Lecce"],
    "monza": ["Monza", "AC Monza"],
    "napoli": ["Napoli", "SSC Napoli"],
    "parma": ["Parma", "Parma Calcio 1913"],
    "pisa": ["Pisa", "AC Pisa 1909"],
    "as-roma": ["Roma", "AS Roma", "A.S. Roma"],
    "salernitana": ["Salernitana", "US Salernitana 1919"],
    "sampdoria": ["Sampdoria", "UC Sampdoria"],
    "sassuolo": ["Sassuolo", "US Sassuolo Calcio"],
    "spal": ["Spal", "SPAL 2013"],
    "spezia": ["Spezia", "Spezia Calcio"],
    "torino": ["Torino", "Torino FC"],
    "udinese": ["Udinese", "Udinese Calcio"],
    "venezia": ["Venezia", "Venezia FC"],

    # ======================== GERMANY ========================
    "augsburg": ["Augsburg", "FC Augsburg"],
    "bayern-munich": ["Bayern München", "FC Bayern München", "Bayern Munich", "FC Bayern", "Bayern"],
    "bielefeld": ["Bielefeld", "Arminia Bielefeld"],
    "bochum": ["Bochum", "VfL Bochum"],
    "borussia-dortmund": ["Borussia Dortmund", "Dortmund", "BVB"],
    "borussia-monchengladbach": ["Borussia Mönchengladbach", "M'gladbach", "Mönchengladbach", "Gladbach", "B. Mönchengladbach"],
    "darmstadt": ["Darmstadt", "SV Darmstadt 98"],
    "eintracht-frankfurt": ["Ein Frankfurt", "Eintracht Frankfurt", "E. Frankfurt"],
    "freiburg": ["Freiburg", "SC Freiburg"],
    "fortuna-dusseldorf": ["Fortuna Dusseldorf", "Fortuna Düsseldorf"],
    "greuther-furth": ["Greuther Furth", "SpVgg Greuther Fürth", "Greuther Fürth"],
    "hamburger-sv": ["Hamburg", "Hamburger SV", "HSV"],
    "hannover-96": ["Hannover", "Hannover 96"],
    "heidenheim": ["Heidenheim", "1. FC Heidenheim 1846", "1. FC Heidenheim"],
    "hertha-berlin": ["Hertha", "Hertha BSC", "Hertha Berlin"],
    "hoffenheim": ["Hoffenheim", "TSG 1899 Hoffenheim", "TSG Hoffenheim"],
    "holstein-kiel": ["Holstein Kiel", "KSV Holstein Kiel"],
    "fc-koln": ["FC Koln", "1. FC Köln", "FC Köln", "Köln"],
    "bayer-leverkusen": ["Leverkusen", "Bayer 04 Leverkusen", "Bayer Leverkusen"],
    "mainz-05": ["Mainz", "1. FSV Mainz 05", "FSV Mainz 05", "Mainz 05"],
    "nurnberg": ["Nurnberg", "1. FC Nürnberg", "Nürnberg"],
    "paderborn": ["Paderborn", "SC Paderborn 07"],
    "rb-leipzig": ["RB Leipzig", "Leipzig", "R.B. Leipzig"],
    "schalke-04": ["Schalke 04", "FC Schalke 04", "Schalke"],
    "st-pauli": ["St Pauli", "FC St. Pauli 1910", "FC St. Pauli"],
    "stuttgart": ["Stuttgart", "VfB Stuttgart"],
    "union-berlin": ["Union Berlin", "1. FC Union Berlin"],
    "werder-bremen": ["Werder Bremen", "SV Werder Bremen"],
    "wolfsburg": ["Wolfsburg", "VfL Wolfsburg"],

    # ======================== FRANCE ========================
    "ajaccio": ["Ajaccio", "AC Ajaccio"],
    "amiens": ["Amiens", "Amiens SC"],
    "angers": ["Angers", "Angers SCO"],
    "auxerre": ["Auxerre", "AJ Auxerre"],
    "bordeaux": ["Bordeaux", "Girondins de Bordeaux"],
    "brest": ["Brest", "Stade Brestois 29"],
    "caen": ["Caen", "SM Caen"],
    "clermont": ["Clermont", "Clermont Foot"],
    "dijon": ["Dijon", "Dijon FCO"],
    "guingamp": ["Guingamp", "EA Guingamp"],
    "le-havre": ["Le Havre", "Le Havre AC"],
    "lens": ["Lens", "Racing Club de Lens", "RC Lens"],
    "lille": ["Lille", "Lille OSC"],
    "lorient": ["Lorient", "FC Lorient"],
    "lyon": ["Lyon", "Olympique Lyonnais", "OL"],
    "marseille": ["Marseille", "Olympique de Marseille", "Olympique Marseille", "OM"],
    "metz": ["Metz", "FC Metz"],
    "monaco": ["Monaco", "AS Monaco FC", "AS Monaco"],
    "montpellier": ["Montpellier", "Montpellier HSC"],
    "nantes": ["Nantes", "FC Nantes"],
    "nice": ["Nice", "OGC Nice"],
    "nimes": ["Nimes", "Nîmes Olympique", "Nîmes"],
    "paris-saint-germain": ["Paris Saint-Germain", "Paris Saint-Germain FC", "PSG", "Paris SG"],
    "paris-fc": ["Paris FC", "Paris"],
    "reims": ["Reims", "Stade de Reims"],
    "rennes": ["Rennes", "Stade Rennais FC 1901", "Stade Rennais"],
    "st-etienne": ["St Etienne", "AS Saint-Étienne", "Saint-Étienne"],
    "strasbourg": ["Strasbourg", "RC Strasbourg Alsace", "RC Strasbourg"],
    "toulouse": ["Toulouse", "Toulouse FC"],
    "troyes": ["Troyes", "ESTAC Troyes"],

    # ======================== NETHERLANDS ========================
    "ajax": ["Ajax", "AFC Ajax"],
    "almere-city": ["Almere City", "Almere City FC"],
    "az-alkmaar": ["AZ Alkmaar", "AZ"],
    "cambuur": ["Cambuur", "SC Cambuur"],
    "den-haag": ["Den Haag", "ADO Den Haag"],
    "emmen": ["Emmen", "FC Emmen"],
    "excelsior": ["Excelsior", "SBV Excelsior"],
    "feyenoord": ["Feyenoord", "Feyenoord Rotterdam"],
    "fortuna-sittard": ["Fortuna Sittard", "For Sittard"],
    "go-ahead-eagles": ["Go Ahead Eagles", "Go Ahead Eagles Deventer"],
    "graafschap": ["Graafschap", "De Graafschap"],
    "groningen": ["Groningen", "FC Groningen"],
    "heerenveen": ["Heerenveen", "SC Heerenveen"],
    "heracles-almelo": ["Heracles", "Heracles Almelo"],
    "nac-breda": ["NAC Breda", "NAC"],
    "nec-nijmegen": ["Nijmegen", "NEC", "NEC Nijmegen"],
    "pec-zwolle": ["Zwolle", "PEC Zwolle"],
    "psv-eindhoven": ["PSV Eindhoven", "PSV"],
    "sparta-rotterdam": ["Sparta Rotterdam", "Sparta"],
    "telstar": ["Telstar", "Telstar 1963"],
    "twente": ["Twente", "FC Twente '65", "FC Twente"],
    "utrecht": ["Utrecht", "FC Utrecht"],
    "vitesse": ["Vitesse", "Vitesse Arnhem"],
    "volendam": ["Volendam", "FC Volendam"],
    "vvv-venlo": ["VVV Venlo", "VVV-Venlo"],
    "waalwijk": ["Waalwijk", "RKC Waalwijk"],
    "willem-ii": ["Willem II", "Willem II Tilburg"],
}

# ============================================================================
# Reverse lookup cache
# ============================================================================
_NAME_TO_ID: Optional[Dict[str, str]] = None


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_for_lookup(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_reverse_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canon_id, variations in TEAM_MAPPINGS.items():
        for name in variations:
            normalized = _normalize_for_lookup(name)
            if normalized:
                mapping[normalized] = canon_id
    return mapping


def _get_reverse_map() -> Dict[str, str]:
    global _NAME_TO_ID
    if _NAME_TO_ID is None:
        _NAME_TO_ID = _build_reverse_map()
    return _NAME_TO_ID


def _fuzzy_match(name: str, threshold: float = 0.80) -> Optional[str]:
    """
    Fuzzy match against known team names.
    Uses token-overlap + SequenceMatcher. Higher threshold to avoid wrong matches.
    Requires a gap from second-best to ensure confidence.
    """
    normalized = _normalize_for_lookup(name)
    if not normalized:
        return None

    reverse_map = _get_reverse_map()
    best_id: Optional[str] = None
    best_score = 0.0
    second_score = 0.0

    norm_tokens = set(normalized.split())

    for known, cid in reverse_map.items():
        known_tokens = set(known.split())
        overlap = len(norm_tokens & known_tokens)
        total = max(len(norm_tokens), len(known_tokens))
        token_score = overlap / max(1, total)

        seq_score = SequenceMatcher(None, normalized, known).ratio()

        # Combined score
        score = 0.4 * token_score + 0.6 * seq_score

        if score > best_score:
            second_score = best_score
            best_score = score
            best_id = cid
        elif score > second_score:
            second_score = score

    # Require confidence AND gap from second-best
    if best_id and best_score >= threshold and (best_score - second_score) >= 0.04:
        return best_id

    return None


def get_canonical_id(raw_name: Optional[str], provider: Optional[str] = None,
                     confidence_threshold: float = 0.80) -> Tuple[Optional[str], bool]:
    """
    Convert any team name to canonical_id.

    Returns:
        (canonical_id, is_exact_match)
    """
    if not raw_name:
        return None, False

    normalized = _normalize_for_lookup(raw_name)
    reverse_map = _get_reverse_map()

    # Exact match
    if normalized in reverse_map:
        return reverse_map[normalized], True

    # Fuzzy match (with higher threshold)
    fuzzy_id = _fuzzy_match(raw_name, threshold=confidence_threshold)
    if fuzzy_id:
        return fuzzy_id, False

    return None, False


def get_canonical_name(raw_name: Optional[str], provider: Optional[str] = None) -> Optional[str]:
    """
    Convert any team name to the canonical display name.
    Returns first (primary) variant from TEAM_MAPPINGS.
    Falls back to cleaned-up original if no match.
    """
    if not raw_name:
        return raw_name

    canonical_id, _ = get_canonical_id(raw_name, provider)
    if canonical_id and canonical_id in TEAM_MAPPINGS:
        variations = TEAM_MAPPINGS[canonical_id]
        if variations:
            return variations[0]
    return raw_name  # Fallback


def register_team_mapping(raw_name: str, canonical_id: str) -> None:
    """Register a new team name mapping dynamically."""
    global _NAME_TO_ID
    if canonical_id not in TEAM_MAPPINGS:
        TEAM_MAPPINGS[canonical_id] = []
    if raw_name not in TEAM_MAPPINGS[canonical_id]:
        TEAM_MAPPINGS[canonical_id].append(raw_name)
    _NAME_TO_ID = None  # Invalidate cache


def ensure_team_mapping_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS team_mappings (
            canonical_id VARCHAR PRIMARY KEY,
            canonical_name VARCHAR NOT NULL,
            provider_names VARCHAR NOT NULL,
            confidence_score DOUBLE DEFAULT 1.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes VARCHAR
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS team_name_lookups (
            raw_name VARCHAR PRIMARY KEY,
            canonical_id VARCHAR NOT NULL,
            confidence DOUBLE,
            lookup_count INT DEFAULT 1,
            last_looked_up TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (canonical_id) REFERENCES team_mappings(canonical_id)
        );
    """)


def sync_mappings_to_db(con: duckdb.DuckDBPyConnection) -> None:
    import json
    ensure_team_mapping_table(con)
    for canonical_id, variations in TEAM_MAPPINGS.items():
        primary_name = variations[0] if variations else canonical_id
        provider_names_json = json.dumps(variations)
        con.execute("""
            INSERT OR REPLACE INTO team_mappings
            (canonical_id, canonical_name, provider_names)
            VALUES (?, ?, ?)
        """, [canonical_id, primary_name, provider_names_json])


def log_team_lookup(con: duckdb.DuckDBPyConnection, raw_name: str, canonical_id: str, confidence: float) -> None:
    try:
        con.execute("""
            INSERT INTO team_name_lookups (raw_name, canonical_id, confidence)
            VALUES (?, ?, ?)
            ON CONFLICT(raw_name) DO UPDATE SET
                lookup_count = lookup_count + 1,
                last_looked_up = CURRENT_TIMESTAMP
        """, [raw_name, canonical_id, confidence])
    except Exception:
        pass


def get_team_match_rate(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> Dict[str, float]:
    """Check what % of teams in database have canonical mappings."""
    try:
        all_teams = con.execute("""
            SELECT DISTINCT team FROM (
                SELECT home_team as team FROM matches WHERE home_team IS NOT NULL
                UNION
                SELECT away_team as team FROM matches WHERE away_team IS NOT NULL
            ) LIMIT 5000
        """).df()

        matched = 0
        unmatched_list = []
        total = len(all_teams)

        for team in all_teams["team"]:
            canonical_id, is_exact = get_canonical_id(team)
            if canonical_id:
                matched += 1
                if verbose:
                    print(f"  ✓ {team} → {canonical_id} ({'exact' if is_exact else 'fuzzy'})")
            else:
                unmatched_list.append(team)
                if verbose:
                    print(f"  ✗ {team} (unmapped)")

        rate = matched / max(1, total)
        if verbose:
            print(f"\nTeam match rate: {matched}/{total} ({rate:.1%})")

        return {"matched": matched, "total": total, "rate": rate, "unmapped": unmatched_list}
    except Exception as e:
        if verbose:
            print(f"Error checking team match rate: {e}")
        return {"matched": 0, "total": 0, "rate": 0.0, "unmapped": []}
