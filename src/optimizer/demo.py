"""Single-vehicle MILP smoke test.

Solves the 24h MILP for one vehicle on two illustrative days:
a grid-event summer afternoon and a calm spring day. Prints a one-line
summary for each and saves a stacked-bar plot of actions + SoC to
``results/milp_demo_<label>.png``.

Run: ``python -m src.optimizer.demo``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONFIG, PHOENIX_TZ
from src.forecasting.forecasters import sample_scenarios_for_vehicle
from src.optimizer.milp import solve_milp


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _build_availability_window(
    availability: pd.DataFrame,
    trips: pd.DataFrame,
    vehicle_id: str,
    asof_ts: pd.Timestamp,
    horizon_h: int,
) -> pd.DataFrame:
    """Slice the full-year availability to a 24h window and attach trip_kwh_this_hour."""
    vehicle_avail = availability[availability["vehicle_id"] == vehicle_id].copy()
    vehicle_avail["timestamp"] = pd.to_datetime(vehicle_avail["timestamp"])
    window = vehicle_avail[
        (vehicle_avail["timestamp"] >= asof_ts)
        & (vehicle_avail["timestamp"] < asof_ts + pd.Timedelta(hours=horizon_h))
    ].copy()
    window = window.sort_values("timestamp").reset_index(drop=True)

    # Attach trip_kwh: concentrated at the trip's departure hour.
    window["trip_kwh_this_hour"] = 0.0
    vtrips = trips[trips["vehicle_id"] == vehicle_id].copy()
    vtrips["depart_ts"] = pd.to_datetime(vtrips["depart_ts"])
    vtrips_in = vtrips[
        (vtrips["depart_ts"] >= asof_ts)
        & (vtrips["depart_ts"] < asof_ts + pd.Timedelta(hours=horizon_h))
    ]
    for _, trip in vtrips_in.iterrows():
        dep_hour = trip["depart_ts"].floor("h")
        matches = window["timestamp"] == dep_hour
        window.loc[matches, "trip_kwh_this_hour"] += trip["kwh"]
    return window[["timestamp", "is_parked", "is_at_home", "trip_kwh_this_hour"]]


def _plot(label: str, asof_ts: pd.Timestamp, result, availability_window: pd.DataFrame) -> Path:
    d = result.decisions.copy()
    d["hour"] = range(len(d))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_act, ax_soc, ax_avail = axes

    width = 0.6
    ax_act.bar(d["hour"], d["charge_kw"], width=width, color="steelblue", label="charge kW")
    ax_act.bar(d["hour"], -d["discharge_kw"], width=width, color="firebrick", label="discharge kW")
    ax_act.bar(d["hour"], d["infer"] * 6.0, width=width * 0.5,
               color="darkorange", alpha=0.7, label="infer (·6kW)")
    ax_act.axhline(0, color="k", lw=0.5)
    ax_act.set_ylabel("kW")
    ax_act.set_title(f"MILP dispatch — vehicle plan starting {label}")
    ax_act.legend(loc="upper right", fontsize=8)
    ax_act.grid(alpha=0.3)

    ax_soc.plot(d["hour"], d["soc_kwh"], marker="o", color="seagreen")
    ax_soc.axhline(CONFIG.battery.reserve_min_kwh, color="red", ls="--", alpha=0.5,
                   label="reserve_min")
    ax_soc.set_ylabel("SoC (kWh)")
    ax_soc.legend(fontsize=8)
    ax_soc.grid(alpha=0.3)

    ax_avail.step(range(len(availability_window)),
                  availability_window["is_at_home"].astype(int),
                  where="post", color="slategray")
    ax_avail.fill_between(range(len(availability_window)),
                          availability_window["is_at_home"].astype(int),
                          step="post", alpha=0.2, color="slategray")
    ax_avail.set_ylabel("is_at_home")
    ax_avail.set_xlabel("hour in horizon")
    ax_avail.set_ylim(-0.1, 1.2)
    ax_avail.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / f"milp_demo_{label}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def run_case(
    label: str,
    asof_ts: pd.Timestamp,
    fleet: pd.DataFrame,
    availability: pd.DataFrame,
    trips: pd.DataFrame,
    n_scenarios: int = 5,
) -> None:
    vehicle = fleet.iloc[0]  # use v000
    vehicle_id = vehicle["vehicle_id"]
    lam = float(vehicle["lambda_miss_usd_per_h"])

    scenarios = sample_scenarios_for_vehicle(
        vehicle_id=vehicle_id, asof_ts=asof_ts, horizon_h=24, n_scenarios=n_scenarios, seed=2026,
    )
    window = _build_availability_window(availability, trips, vehicle_id, asof_ts, horizon_h=24)

    # Start the vehicle at mid-range SoC (50% of flexible band above reserve)
    soc0 = CONFIG.battery.reserve_max_kwh + 0.5 * CONFIG.battery.flexible_max_kwh
    result = solve_milp(
        soc_init_kwh=soc0,
        lambda_miss_usd_per_h=lam,
        scenarios=scenarios,
        availability_window=window,
    )

    print(
        f"[{label}] status={result.status}  "
        f"obj=${result.objective_value:.2f}  "
        f"solve={result.solve_time_s*1000:.0f}ms  "
        f"first-stage charge={result.first_stage['charge_kw']:.2f}kW  "
        f"discharge={result.first_stage['discharge_kw']:.2f}kW  "
        f"infer={result.first_stage['infer']}"
    )
    total_charge = result.decisions["charge_kw"].sum()
    total_discharge = result.decisions["discharge_kw"].sum()
    total_infer_h = result.decisions["infer"].sum()
    print(
        f"           24h: charged {total_charge:.1f} kWh, discharged {total_discharge:.1f} kWh, "
        f"inference {total_infer_h:.1f} h. end SoC = {result.decisions['soc_kwh'].iloc[-1]:.1f} kWh."
    )
    _plot(label, asof_ts, result, window)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fleet = pd.read_parquet(DATA_DIR / "fleet.parquet")
    availability = pd.read_parquet(DATA_DIR / "availability.parquet")
    trips = pd.read_parquet(DATA_DIR / "trips.parquet")

    # Two test days
    cases = [
        # known summer event day — check opt exploits the LMP spike
        ("event_summer_day", pd.Timestamp("2026-08-23 00:00", tz=PHOENIX_TZ)),
        # calm spring day with no events — check conservative behavior
        ("calm_spring_day", pd.Timestamp("2026-04-10 00:00", tz=PHOENIX_TZ)),
        # winter morning event day
        ("event_winter_day", pd.Timestamp("2026-02-01 00:00", tz=PHOENIX_TZ)),
    ]
    for label, ts in cases:
        run_case(label, ts, fleet, availability, trips)


if __name__ == "__main__":
    main()
