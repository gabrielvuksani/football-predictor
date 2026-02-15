from __future__ import annotations

import io
from dataclasses import dataclass
import httpx
import pandas as pd

# football-data.co.uk historical CSV pattern examples: /mmz4281/{season}/E0.csv :contentReference[oaicite:2]{index=2}
BASE = "https://www.football-data.co.uk/mmz4281"

@dataclass(frozen=True)
class FDCUKDivision:
    div: str          # e.g., E0, D1, I1, SP1, F1, N1
    competition: str  # e.g., PL, BL1, SA, PD, FL1, DED

DIV_MAP = {
    "PL":  FDCUKDivision("E0",  "PL"),
    "PD":  FDCUKDivision("SP1", "PD"),
    "SA":  FDCUKDivision("I1",  "SA"),
    "BL1": FDCUKDivision("D1",  "BL1"),
    "FL1": FDCUKDivision("F1",  "FL1"),
    "DED": FDCUKDivision("N1",  "DED"),
}

def season_codes_last_n(n: int, include_current: bool = True) -> list[str]:
    """
    Returns season folder codes like 2526, 2425, ...
    Uses current year (UTC) heuristic.
    """
    import datetime
    y = datetime.datetime.now(datetime.timezone.utc).year
    # football season spans; assume current season started previous year if before July
    m = datetime.datetime.now(datetime.timezone.utc).month
    start_year = y if m >= 7 else y - 1
    if not include_current:
        start_year -= 1

    codes = []
    for k in range(n):
        a = start_year - k
        b = a + 1
        codes.append(f"{str(a)[2:]}{str(b)[2:]}")
    return codes

def download_division_csv(season_code: str, div_code: str) -> pd.DataFrame:
    url = f"{BASE}/{season_code}/{div_code}.csv"
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout, headers={"User-Agent": "footy-predictor/0.1"}) as c:
        r = c.get(url)
        r.raise_for_status()
        # some files contain non-utf8 characters; latin1 is safe fallback
        text = r.content.decode("utf-8", errors="ignore")
        if "Div,Date" not in text[:200]:
            text = r.content.decode("latin-1", errors="ignore")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")
