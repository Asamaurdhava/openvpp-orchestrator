"""Download real Phoenix hourly temperature for the simulation year.

Why
---
The backtest uses a synthetic grid-event generator. To show judges that our
events line up with *real* Arizona stress conditions, we overlay real Phoenix
weather on top of the simulated event windows. Hot afternoons in July/August
line up with real NOAA records — and that lends credibility to the synthetic
trace without replacing it.

Source: Open-Meteo Archive API (no signup, no token).
  Station proxy: Phoenix Sky Harbor coordinates (33.45°N, 112.07°W).
  Year:          2026 (matches SIM_START in src/config.py; if 2026 data is
                 not yet available, we fall back to the latest complete year).

Output: data/real_phoenix_temp.parquet
Columns: timestamp (tz-aware, America/Phoenix), temp_f, temp_c
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real_phoenix_temp.parquet"

PHX_LAT = 33.4484
PHX_LON = -112.0740
TZ = "America/Phoenix"


def _fetch(year: int) -> pd.DataFrame | None:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={PHX_LAT}&longitude={PHX_LON}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit"
        f"&timezone={TZ}"
    )
    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as e:
        print(f"  year {year}: network error — {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  year {year}: HTTP {r.status_code}", file=sys.stderr)
        return None
    payload = r.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time")
    temps = hourly.get("temperature_2m")
    if not times or not temps:
        print(f"  year {year}: empty response", file=sys.stderr)
        return None
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times).tz_localize(TZ),
            "temp_f": temps,
        }
    )
    df["temp_c"] = (df["temp_f"] - 32) * 5 / 9
    return df


def main() -> None:
    # Prefer the simulation year; fall back to the previous complete year if the
    # API doesn't have full coverage yet (it usually lags by ~5 days).
    for year in (2026, 2025, 2024):
        print(f"fetching Phoenix hourly temp for {year}…")
        df = _fetch(year)
        if df is not None and len(df) > 8000:  # full year ≈ 8760 rows
            OUT.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUT, index=False)
            n = len(df)
            tmax = df["temp_f"].max()
            tmin = df["temp_f"].min()
            print(f"  wrote {OUT.name}  ({n} rows · range {tmin:.0f}°F → {tmax:.0f}°F)")
            return
        print(f"  year {year}: incomplete ({0 if df is None else len(df)} rows) — trying next")

    print("no usable year available — real-data overlay will stay disabled.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
