"""End-of-Phase-1 sanity check: print summary stats + save 4 diagnostic plots.

Reads all Phase 1 parquets from ``data/`` and writes PNGs to ``results/``:

* ``sanity_lmp_hour_of_day.png``
* ``sanity_grid_events_timeline.png``
* ``sanity_fleet_trip_distribution.png``
* ``sanity_inference_participation.png``

Fails loudly if the synthetic traces violate the challenge's published
ranges (from ``docs/assumptions.md``).

Run directly: ``python -m src.sanity_check``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — save PNGs only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONFIG


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _assert_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise AssertionError(f"{name} = {value:.3f} outside [{lo}, {hi}]")
    print(f"  ok  {name} = {value:.3f}  (expected {lo}–{hi})")


def check_fleet(fleet: pd.DataFrame, trips: pd.DataFrame) -> None:
    fp = CONFIG.fleet
    drv = CONFIG.driving
    print("\n== fleet ==")
    assert len(fleet) == fp.n_vehicles, f"expected {fp.n_vehicles} vehicles, got {len(fleet)}"
    n_private = (fleet["ownership"] == "private").sum()
    n_fleet = (fleet["ownership"] == "fleet").sum()
    print(f"  ok  {n_private} private + {n_fleet} fleet = {len(fleet)} total")

    # Trips should mostly land in challenge range.
    mean_kwh_per_weekday = (
        trips[trips["trip_type"] == "commute"]
        .groupby(trips[trips["trip_type"] == "commute"]["trip_date"])["kwh"]
        .sum()
        .mean()
    )
    # Per-vehicle: daily driving consumption should be 25–40 kWh on a commute day.
    per_vehicle_commute_mean = (
        trips[trips["trip_type"] == "commute"].groupby("vehicle_id")["kwh"].mean()
    )
    _assert_range(
        "mean commute trip kWh (per vehicle)",
        per_vehicle_commute_mean.mean(),
        drv.daily_kwh_min * 0.9,
        drv.daily_kwh_max * 1.1,
    )


def check_availability(availability: pd.DataFrame) -> None:
    print("\n== availability ==")
    parked_frac = availability["is_parked"].mean()
    home_frac = availability["is_at_home"].mean()
    # Idle time 18–22 h/day = 75–92% of hours.
    _assert_range("parked fraction", parked_frac, 0.70, 0.95)
    _assert_range("at-home fraction", home_frac, 0.70, 0.95)


def check_grid_events(grid: pd.DataFrame) -> None:
    print("\n== grid events ==")
    gp = CONFIG.grid
    n_events = grid["event_id"].dropna().unique().size
    assert n_events == gp.n_summer + gp.n_winter, (
        f"expected {gp.n_summer + gp.n_winter} events, got {n_events}"
    )
    print(f"  ok  {n_events} events total")

    # Duration check
    dur = grid[grid["is_event"]].groupby("event_id", observed=True).size()
    assert dur.between(gp.duration_h_min, gp.duration_h_max).all(), (
        f"event durations out of range: {dur.to_dict()}"
    )
    print(f"  ok  event durations ∈ [{dur.min()}, {dur.max()}] h")


def check_lmp(lmp: pd.DataFrame) -> None:
    print("\n== lmp ==")
    lp = CONFIG.lmp
    # Challenge range for peak event pricing is $0.20–$0.50; no clamp should bite in v1.
    _assert_range("min LMP", lmp["lmp_usd_per_kwh"].min(), lp.clamp_min_usd, 0.10)
    _assert_range("max LMP", lmp["lmp_usd_per_kwh"].max(), 0.30, lp.clamp_max_usd)
    by_hour = lmp.assign(h=lmp["timestamp"].dt.hour).groupby("h")["lmp_usd_per_kwh"].mean()
    onpeak_mean = by_hour.loc[list(lp.onpeak_hours)].mean()
    offpeak_mean = by_hour.drop(list(lp.onpeak_hours)).mean()
    assert onpeak_mean > offpeak_mean, f"on-peak {onpeak_mean:.3f} not > off-peak {offpeak_mean:.3f}"
    print(f"  ok  on-peak mean ${onpeak_mean:.3f} > off-peak mean ${offpeak_mean:.3f}")


def check_inference(inference: pd.DataFrame) -> None:
    print("\n== inference_demand ==")
    ip = CONFIG.inference
    days_with_demand = (
        inference[inference["demand_available"]]
        .assign(d=lambda x: x["timestamp"].dt.date)
        .groupby("vehicle_id")["d"]
        .nunique()
    )
    # Challenge target: 150–250 days/year. Allow a bit of slack.
    _assert_range("mean days with demand / vehicle", days_with_demand.mean(), 120, 280)
    active = inference[inference["demand_available"]]
    _assert_range("active price mean", active["revenue_usd_per_kwh"].mean(),
                  ip.revenue_min_usd_per_kwh, ip.revenue_max_usd_per_kwh)
    _assert_range("active power mean", active["power_kw"].mean(),
                  ip.power_min_kw, ip.power_max_kw)


def plot_lmp(lmp: pd.DataFrame) -> None:
    by_hour = lmp.assign(h=lmp["timestamp"].dt.hour).groupby("h")["lmp_usd_per_kwh"]
    mean = by_hour.mean()
    p10 = by_hour.quantile(0.10)
    p90 = by_hour.quantile(0.90)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean.index, mean.values, label="mean", color="steelblue", lw=2)
    ax.fill_between(mean.index, p10.values, p90.values, alpha=0.25, color="steelblue",
                    label="10–90 pct")
    ax.set_xlabel("hour of day (Phoenix local)")
    ax.set_ylabel("LMP ($/kWh)")
    ax.set_title("LMP diurnal pattern — whole year")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sanity_lmp_hour_of_day.png", dpi=120)
    plt.close(fig)


def plot_grid_events(grid: pd.DataFrame) -> None:
    ev = grid[grid["is_event"]].copy()
    ev["day"] = ev["timestamp"].dt.dayofyear
    ev["hour"] = ev["timestamp"].dt.hour

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(ev["day"], ev["hour"], c=ev["severity"], cmap="Reds", s=40, edgecolor="k")
    ax.set_xlabel("day of year")
    ax.set_ylabel("hour of day")
    ax.set_title("Grid stress events — 6 summer afternoons + 4 winter mornings")
    ax.set_yticks(range(0, 24, 3))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sanity_grid_events_timeline.png", dpi=120)
    plt.close(fig)


def plot_trips(trips: pd.DataFrame) -> None:
    per_vehicle = trips[trips["trip_type"] == "commute"].groupby("vehicle_id")["kwh"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_vehicle, bins=25, color="seagreen", edgecolor="k")
    ax.axvline(CONFIG.driving.daily_kwh_min, ls="--", color="k", alpha=0.5, label="challenge min")
    ax.axvline(CONFIG.driving.daily_kwh_max, ls="--", color="k", alpha=0.5, label="challenge max")
    ax.set_xlabel("mean weekday trip kWh (per vehicle)")
    ax.set_ylabel("count")
    ax.set_title("Fleet trip energy distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sanity_fleet_trip_distribution.png", dpi=120)
    plt.close(fig)


def plot_inference(inference: pd.DataFrame) -> None:
    days = (
        inference[inference["demand_available"]]
        .assign(d=lambda x: x["timestamp"].dt.date)
        .groupby("vehicle_id")["d"]
        .nunique()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(days, bins=25, color="darkorange", edgecolor="k")
    ax.axvline(150, ls="--", color="k", alpha=0.5, label="challenge min (150 d/yr)")
    ax.axvline(250, ls="--", color="k", alpha=0.5, label="challenge max (250 d/yr)")
    ax.set_xlabel("days with inference demand / vehicle / year")
    ax.set_ylabel("count")
    ax.set_title("Inference participation distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sanity_inference_participation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fleet = pd.read_parquet(DATA_DIR / "fleet.parquet")
    trips = pd.read_parquet(DATA_DIR / "trips.parquet")
    availability = pd.read_parquet(DATA_DIR / "availability.parquet")
    grid = pd.read_parquet(DATA_DIR / "grid_events.parquet")
    lmp = pd.read_parquet(DATA_DIR / "lmp.parquet")
    inference = pd.read_parquet(DATA_DIR / "inference_demand.parquet")

    check_fleet(fleet, trips)
    check_availability(availability)
    check_grid_events(grid)
    check_lmp(lmp)
    check_inference(inference)

    plot_lmp(lmp)
    plot_grid_events(grid)
    plot_trips(trips)
    plot_inference(inference)

    print(f"\nplots written to {RESULTS_DIR}")
    print("all sanity checks passed")


if __name__ == "__main__":
    main()
