from __future__ import annotations
import io
import httpx
import pandas as pd

from footy.providers.ratelimit import retry_request

URL = "https://www.football-data.co.uk/fixtures.csv"

def download_fixtures() -> pd.DataFrame:
    timeout = httpx.Timeout(30.0, connect=10.0)

    def _do_request():
        with httpx.Client(timeout=timeout, headers={"User-Agent": "footy-predictor/0.1"}) as c:
            r = c.get(URL)
            r.raise_for_status()
            return r

    r = retry_request(_do_request, label="fdcuk/fixtures")
    text = r.content.decode("utf-8", errors="ignore")
    if "Div,Date" not in text[:200]:
        text = r.content.decode("latin-1", errors="ignore")
    return pd.read_csv(io.StringIO(text))
