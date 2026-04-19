"""Arizona grid stress events for the simulation year.

Produces ``data/grid_events.parquet`` with one row per hour of the year:
``timestamp``, ``is_event``, ``event_id``, ``severity``.

Challenge parameters: 10 events/year total, 6 summer afternoons + 4 winter
mornings, 3–5h duration each. See ``docs/assumptions.md §1, §2.8``.

Run directly: ``python -m src.signals.grid_events``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, SIM_HORIZON_DAYS, SIM_START, PHOENIX_TZ, SIM_YEAR, seed_for


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _pick_event_days(
    rng: np.random.Generator, months: tuple[int, ...], n_events: int
) -> list[pd.Timestamp]:
    """Pick n distinct weekdays (mostly) within the given months."""
    candidates: list[pd.Timestamp] = []
    for m in months:
        year = SIM_YEAR + (1 if m == 1 and months == CONFIG.grid.winter_months else 0)
        # winter_months = (12, 1, 2); month 12 is in SIM_YEAR, months 1 & 2 are in SIM_YEAR+1.
        # but grid events are drawn within SIM_YEAR only, so cap at SIM_YEAR.
        year = SIM_YEAR
        if m == 1 or m == 2:
            # Pull from the January/February of SIM_YEAR (early in the sim).
            pass
        ts_start = pd.Timestamp(year=year, month=m, day=1, tz=PHOENIX_TZ)
        next_month = 1 if m == 12 else m + 1
        next_year = year + 1 if m == 12 else year
        ts_end = pd.Timestamp(year=next_year, month=next_month, day=1, tz=PHOENIX_TZ)
        days = pd.date_range(ts_start, ts_end - pd.Timedelta(days=1), freq="D")
        # Restrict to days that fall inside the simulation horizon.
        sim_end = pd.Timestamp(SIM_START, tz=PHOENIX_TZ) + pd.Timedelta(days=SIM_HORIZON_DAYS)
        days = days[(days >= pd.Timestamp(SIM_START, tz=PHOENIX_TZ)) & (days < sim_end)]
        candidates.extend(days.tolist())

    if len(candidates) < n_events:
        raise ValueError(f"Only {len(candidates)} candidate days for {n_events} events")

    picks = rng.choice(len(candidates), size=n_events, replace=False)
    return sorted(candidates[i] for i in picks)


def _build_event_hours(
    rng: np.random.Generator,
    days: list[pd.Timestamp],
    start_hour: int,
    end_hour: int,
    event_id_start: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    gp = CONFIG.grid
    for offset, d in enumerate(days):
        eid = event_id_start + offset
        hour_start = int(rng.integers(start_hour, end_hour + 1))
        duration = int(rng.integers(gp.duration_h_min, gp.duration_h_max + 1))
        severity = float(rng.uniform(0.5, 1.0))
        for h in range(duration):
            ts = d + pd.Timedelta(hours=hour_start + h)
            rows.append(
                {
                    "timestamp": ts,
                    "event_id": eid,
                    "severity": severity,
                }
            )
    return pd.DataFrame.from_records(rows)


def generate(seed: int | None = None) -> pd.DataFrame:
    if seed is None:
        seed = seed_for("signals.grid_events")
    rng = np.random.default_rng(seed)
    gp = CONFIG.grid

    summer_days = _pick_event_days(rng, gp.summer_months, gp.n_summer)
    winter_days = _pick_event_days(rng, gp.winter_months, gp.n_winter)

    summer_events = _build_event_hours(
        rng, summer_days, gp.summer_hour_start, gp.summer_hour_end, event_id_start=0
    )
    winter_events = _build_event_hours(
        rng, winter_days, gp.winter_hour_start, gp.winter_hour_end, event_id_start=gp.n_summer
    )
    events = pd.concat([summer_events, winter_events], ignore_index=True)

    hours = pd.date_range(
        SIM_START, periods=SIM_HORIZON_DAYS * 24, freq="h", tz=PHOENIX_TZ
    )
    out = pd.DataFrame({"timestamp": hours, "is_event": False, "event_id": pd.NA, "severity": 0.0})
    out = out.set_index("timestamp")
    events = events.set_index("timestamp")
    overlap = events.index.intersection(out.index)
    out.loc[overlap, "is_event"] = True
    out.loc[overlap, "event_id"] = events.loc[overlap, "event_id"].astype("Int64")
    out.loc[overlap, "severity"] = events.loc[overlap, "severity"].astype(float)
    out = out.reset_index()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_DIR / "grid_events.parquet", index=False)
    return out


def main() -> None:
    df = generate()
    gp = CONFIG.grid
    n_event_hours = int(df["is_event"].sum())
    n_events = df["event_id"].dropna().unique().size
    print(f"grid_events: {n_events} events spanning {n_event_hours} hours "
          f"(expected {gp.n_summer + gp.n_winter} events)")
    summary = (
        df[df["is_event"]]
        .groupby("event_id", observed=True)
        .agg(start=("timestamp", "min"), end=("timestamp", "max"), severity=("severity", "first"))
        .sort_values("start")
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
