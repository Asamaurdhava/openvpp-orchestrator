"""Per-vehicle inference demand signal.

Emits ``data/inference_demand.parquet`` with one row per vehicle × hour:
``vehicle_id``, ``timestamp``, ``demand_available``, ``power_kw``,
``revenue_usd_per_kwh``.

Model: for each vehicle, draw session starts from a Poisson process with a
time-varying rate (higher daytime, lower overnight). Each session persists
for 4–10h at a fixed power (2–8 kW) and revenue price ($0.20–$0.60/kWh-eq)
drawn once per session. Sessions never overlap for the same vehicle.

The optimizer decides whether to *accept* demand — this signal only says
whether work is *offered*.

Run directly: ``python -m src.signals.inference_demand``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, SIM_HORIZON_DAYS, SIM_START, PHOENIX_TZ, seed_for


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _sample_sessions_for_vehicle(
    rng: np.random.Generator,
    vehicle_id: str,
    hours: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return long-form sessions: one row per (vehicle × active hour)."""
    ip = CONFIG.inference
    n = len(hours)
    hr_of_day = hours.hour.to_numpy()
    rates = np.where(np.isin(hr_of_day, ip.daytime_hours), ip.daytime_rate_per_h, ip.overnight_rate_per_h)

    # Per-hour Bernoulli on the thinned Poisson.
    starts = rng.random(n) < rates

    occupied_until = -1
    active = np.zeros(n, dtype=bool)
    power = np.zeros(n)
    price = np.zeros(n)

    for i in range(n):
        if i <= occupied_until:
            continue
        if starts[i]:
            duration = int(rng.integers(ip.session_duration_min_h, ip.session_duration_max_h + 1))
            sess_power = float(rng.uniform(ip.power_min_kw, ip.power_max_kw))
            sess_price = float(rng.uniform(ip.revenue_min_usd_per_kwh, ip.revenue_max_usd_per_kwh))
            end = min(n - 1, i + duration - 1)
            active[i : end + 1] = True
            power[i : end + 1] = sess_power
            price[i : end + 1] = sess_price
            occupied_until = end

    return pd.DataFrame(
        {
            "vehicle_id": vehicle_id,
            "timestamp": hours,
            "demand_available": active,
            "power_kw": power,
            "revenue_usd_per_kwh": price,
        }
    )


def generate(seed: int | None = None) -> pd.DataFrame:
    if seed is None:
        seed = seed_for("signals.inference_demand")
    rng = np.random.default_rng(seed)

    fleet_path = DATA_DIR / "fleet.parquet"
    if not fleet_path.exists():
        raise FileNotFoundError(
            "fleet.parquet missing — run `python -m src.fleet.generator` first."
        )
    fleet = pd.read_parquet(fleet_path)

    hours = pd.date_range(
        SIM_START, periods=SIM_HORIZON_DAYS * 24, freq="h", tz=PHOENIX_TZ
    )

    frames = []
    for vid in fleet["vehicle_id"]:
        # Per-vehicle stream so each vehicle's trace is independent but seeded.
        v_seed = (seed + hash(vid)) & 0x7FFFFFFF
        v_rng = np.random.default_rng(v_seed)
        frames.append(_sample_sessions_for_vehicle(v_rng, vid, hours))

    out = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_DIR / "inference_demand.parquet", index=False)
    return out


def main() -> None:
    df = generate()
    n_vehicles = df["vehicle_id"].nunique()
    active_hours_per_vehicle = df.groupby("vehicle_id")["demand_available"].sum()
    days_with_demand = (
        df[df["demand_available"]]
        .assign(d=lambda x: x["timestamp"].dt.date)
        .groupby("vehicle_id")["d"]
        .nunique()
        .reindex(df["vehicle_id"].unique(), fill_value=0)
    )
    active_prices = df.loc[df["demand_available"], "revenue_usd_per_kwh"]
    active_powers = df.loc[df["demand_available"], "power_kw"]
    print(f"inference_demand: {len(df)} rows over {n_vehicles} vehicles")
    print(f"  active hours / vehicle: mean={active_hours_per_vehicle.mean():.0f}  "
          f"median={active_hours_per_vehicle.median():.0f}")
    print(f"  days with demand / vehicle: mean={days_with_demand.mean():.1f}  "
          f"(challenge target 150–250)")
    print(f"  active price: mean=${active_prices.mean():.3f}  "
          f"min=${active_prices.min():.3f}  max=${active_prices.max():.3f}")
    print(f"  active power: mean={active_powers.mean():.2f} kW  "
          f"range [{active_powers.min():.2f}, {active_powers.max():.2f}]")


if __name__ == "__main__":
    main()
