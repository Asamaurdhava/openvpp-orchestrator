"""Synthetic 100-vehicle Phoenix fleet.

Produces three parquet artifacts under ``data/``:

1. ``fleet.parquet`` — static per-vehicle attributes
2. ``trips.parquet`` — one row per trip (depart_ts, return_ts, kwh)
3. ``availability.parquet`` — per-vehicle hourly (is_parked, is_at_home) flags

Run directly: ``python -m src.fleet.generator``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, SIM_HORIZON_DAYS, SIM_START, PHOENIX_TZ, seed_for


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _sample_fleet(rng: np.random.Generator) -> pd.DataFrame:
    fp = CONFIG.fleet
    drv = CONFIG.driving

    n = fp.n_vehicles
    n_private = int(round(n * fp.private_share))
    ownership = np.array(["private"] * n_private + ["fleet"] * (n - n_private))
    rng.shuffle(ownership)

    home_lat = rng.uniform(fp.phoenix_lat_min, fp.phoenix_lat_max, n)
    home_lon = rng.uniform(fp.phoenix_lon_min, fp.phoenix_lon_max, n)
    is_wfh = rng.random(n) < drv.wfh_probability
    work_lat = np.where(is_wfh, home_lat, rng.uniform(fp.phoenix_lat_min, fp.phoenix_lat_max, n))
    work_lon = np.where(is_wfh, home_lon, rng.uniform(fp.phoenix_lon_min, fp.phoenix_lon_max, n))

    depart = rng.normal(drv.weekday_depart_mean_hour, drv.weekday_depart_sd_hour, n).clip(5.5, 10.5)
    returns = rng.normal(drv.weekday_return_mean_hour, drv.weekday_return_sd_hour, n).clip(14.0, 21.0)
    trip_kwh_mean = rng.uniform(drv.daily_kwh_min, drv.daily_kwh_max, n)

    lam = np.where(
        ownership == "fleet",
        fp.lambda_miss_fleet_usd_per_h,
        fp.lambda_miss_private_usd_per_h,
    )

    return pd.DataFrame(
        {
            "vehicle_id": [f"v{i:03d}" for i in range(n)],
            "ownership": ownership,
            "is_wfh": is_wfh,
            "home_lat": home_lat,
            "home_lon": home_lon,
            "work_lat": work_lat,
            "work_lon": work_lon,
            "weekday_depart_hour": depart,
            "weekday_return_hour": returns,
            "trip_kwh_mean": trip_kwh_mean,
            "lambda_miss_usd_per_h": lam,
        }
    )


def _sample_trips(fleet: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    drv = CONFIG.driving
    records: list[dict] = []

    dates = pd.date_range(SIM_START, periods=SIM_HORIZON_DAYS, freq="D", tz=PHOENIX_TZ)

    for _, v in fleet.iterrows():
        for d in dates:
            is_weekday = d.dayofweek < 5
            if is_weekday and not v["is_wfh"]:
                depart_h = float(rng.normal(v["weekday_depart_hour"], 0.25))
                return_h = float(rng.normal(v["weekday_return_hour"], 0.35))
                depart_h = max(5.5, min(10.5, depart_h))
                return_h = max(depart_h + 4.0, min(21.5, return_h))
                kwh = float(rng.normal(v["trip_kwh_mean"], 3.0))
                kwh = max(drv.daily_kwh_min * 0.6, min(drv.daily_kwh_max * 1.1, kwh))
                records.append(
                    {
                        "vehicle_id": v["vehicle_id"],
                        "trip_date": d.date(),
                        "depart_ts": d + pd.Timedelta(hours=depart_h),
                        "return_ts": d + pd.Timedelta(hours=return_h),
                        "kwh": kwh,
                        "trip_type": "commute",
                    }
                )
            elif not is_weekday and rng.random() < drv.weekend_trip_probability:
                depart_h = float(rng.uniform(9.0, 17.0))
                duration = float(rng.uniform(1.0, 3.5))
                kwh = float(abs(rng.normal(drv.weekend_trip_kwh_mean, drv.weekend_trip_kwh_sd)))
                records.append(
                    {
                        "vehicle_id": v["vehicle_id"],
                        "trip_date": d.date(),
                        "depart_ts": d + pd.Timedelta(hours=depart_h),
                        "return_ts": d + pd.Timedelta(hours=depart_h + duration),
                        "kwh": kwh,
                        "trip_type": "weekend",
                    }
                )

    return pd.DataFrame.from_records(records)


def _build_availability(fleet: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """For each vehicle × hour, mark is_parked and is_at_home.

    Implemented with numpy 2D boolean arrays for speed: O(n_trips + n_vehicles·n_hours)
    instead of pandas .loc assignments per trip.
    """
    n_vehicles = len(fleet)
    n_hours = SIM_HORIZON_DAYS * 24

    vid_to_idx = {vid: i for i, vid in enumerate(fleet["vehicle_id"])}
    hour_start = pd.Timestamp(SIM_START, tz=PHOENIX_TZ)
    hours = pd.date_range(hour_start, periods=n_hours, freq="h", tz=PHOENIX_TZ)

    is_parked = np.ones((n_vehicles, n_hours), dtype=bool)
    is_at_home = np.ones((n_vehicles, n_hours), dtype=bool)

    if not trips.empty:
        hour_start_np = np.datetime64(hour_start.tz_convert("UTC").tz_localize(None), "ns")
        depart_ns = trips["depart_ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
        return_ns = trips["return_ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")

        ns_per_hour = np.timedelta64(3600, "s").astype("timedelta64[ns]").astype(np.int64)
        # ceil(depart) to next hour boundary so the trip starts on an integer hour.
        depart_hours = np.ceil((depart_ns - hour_start_np).astype(np.int64) / ns_per_hour).astype(np.int64)
        return_hours = np.ceil((return_ns - hour_start_np).astype(np.int64) / ns_per_hour).astype(np.int64)
        depart_hours = np.clip(depart_hours, 0, n_hours)
        return_hours = np.clip(return_hours, 0, n_hours)

        v_indices = trips["vehicle_id"].map(vid_to_idx).to_numpy()

        for vi, h0, h1 in zip(v_indices, depart_hours, return_hours):
            if h1 > h0:
                is_parked[vi, h0:h1] = False
                is_at_home[vi, h0:h1] = False

    vids = fleet["vehicle_id"].to_numpy()
    return pd.DataFrame(
        {
            "vehicle_id": np.repeat(vids, n_hours),
            "timestamp": np.tile(hours, n_vehicles),
            "is_parked": is_parked.reshape(-1),
            "is_at_home": is_at_home.reshape(-1),
        }
    )


def generate(seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seed is None:
        seed = seed_for("fleet.generator")
    rng = np.random.default_rng(seed)

    fleet = _sample_fleet(rng)
    trips = _sample_trips(fleet, rng)
    availability = _build_availability(fleet, trips)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fleet.to_parquet(DATA_DIR / "fleet.parquet", index=False)
    trips.to_parquet(DATA_DIR / "trips.parquet", index=False)
    availability.to_parquet(DATA_DIR / "availability.parquet", index=False)

    return fleet, trips, availability


def main() -> None:
    fleet, trips, availability = generate()
    print(f"fleet: {len(fleet)} vehicles  ({(fleet['ownership']=='private').sum()} private, "
          f"{(fleet['ownership']=='fleet').sum()} fleet)")
    print(f"trips: {len(trips)} trips over {SIM_HORIZON_DAYS} days "
          f"(mean trips/vehicle = {len(trips)/len(fleet):.1f})")
    parked_pct = availability["is_parked"].mean() * 100
    home_pct = availability["is_at_home"].mean() * 100
    print(f"availability: parked {parked_pct:.1f}% of hours, at-home {home_pct:.1f}% of hours")


if __name__ == "__main__":
    main()
