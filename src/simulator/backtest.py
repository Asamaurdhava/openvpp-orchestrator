"""One-year fleet backtest across all 5 strategies.

Runs each strategy on the first ``n_vehicles`` members of the fleet in
parallel (process pool), saves per-strategy decisions as parquet, and
computes the headline metrics:

* total revenue per vehicle
* V2G revenue (discharge × LMP during event hours)
* inference revenue
* charging cost
* mobility reliability (fraction of hours with shortfall ≤ 0.5 kWh)
* MWh delivered during stress events

Outputs:

* ``results/backtest_decisions_<strategy>.parquet``
* ``results/backtest_metrics.csv`` (rows = strategy × vehicle)
* ``results/backtest_summary.csv`` (rows = strategy, aggregated)
* ``results/backtest_comparison.png``

Run: ``python -m src.simulator.backtest [--n-vehicles 20]``.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CONFIG, PHOENIX_TZ, SIM_HORIZON_DAYS, SIM_START


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Pool-global traces — loaded once per worker.
_TRACES = None


def _worker_init() -> None:
    """Load traces once per worker process."""
    global _TRACES
    from src.optimizer.rolling_horizon import ActualTraces
    _TRACES = ActualTraces.load()


def _worker_task(args: tuple) -> pd.DataFrame:
    vehicle_dict, strategy_name, start, end = args
    from src.simulator.baselines import STRATEGIES
    fn = STRATEGIES[strategy_name]
    vehicle = pd.Series(vehicle_dict)
    return fn(vehicle=vehicle, traces=_TRACES, start=start, end=end)


def _run_strategy(
    strategy_name: str,
    fleet: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_workers: int,
) -> pd.DataFrame:
    """Run one strategy across the fleet and return a concatenated DataFrame."""
    records = [
        (row.to_dict(), strategy_name, start, end) for _, row in fleet.iterrows()
    ]
    t0 = time.perf_counter()
    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        frames: list[pd.DataFrame] = []
        for df in tqdm(
            pool.imap_unordered(_worker_task, records),
            total=len(records),
            desc=f"  {strategy_name}",
            leave=False,
        ):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    t = time.perf_counter() - t0
    print(f"  {strategy_name}: {len(fleet)} vehicles, wall {t:.0f}s "
          f"({t/len(fleet):.1f}s/vehicle)")
    return out


def _compute_metrics(decisions: pd.DataFrame, grid: pd.DataFrame, lmp: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle metrics from a single-strategy decision DataFrame."""
    d = decisions.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])

    grid_map = grid.set_index("timestamp")["is_event"].astype(bool)
    lmp_map = lmp.set_index("timestamp")["lmp_usd_per_kwh"]

    d["is_event"] = d["timestamp"].map(grid_map).fillna(False).astype(bool)
    d["lmp"] = d["timestamp"].map(lmp_map)

    d["v2g_revenue"] = d["discharge_kw"] * d["lmp"]
    d["charging_cost"] = d["charge_kw"] * d["lmp"]

    # Inference revenue: requires merging inference power+price
    inf = inference[["vehicle_id", "timestamp", "power_kw", "revenue_usd_per_kwh", "demand_available"]].copy()
    inf["timestamp"] = pd.to_datetime(inf["timestamp"])
    d = d.merge(inf, on=["vehicle_id", "timestamp"], how="left")
    d["infer_revenue"] = d["infer_active"] * d["power_kw"].fillna(0) * d["revenue_usd_per_kwh"].fillna(0)

    reserve = 0.5 * (CONFIG.battery.reserve_min_kwh + CONFIG.battery.reserve_max_kwh)

    per_vehicle = (
        d.groupby("vehicle_id")
        .agg(
            total_revenue_usd=("revenue_usd", "sum"),
            v2g_revenue_usd=("v2g_revenue", "sum"),
            inference_revenue_usd=("infer_revenue", "sum"),
            charging_cost_usd=("charging_cost", "sum"),
            total_charge_kwh=("charge_kw", "sum"),
            total_discharge_kwh=("discharge_kw", "sum"),
            infer_hours=("infer_active", "sum"),
            mwh_in_events=("discharge_kw", lambda x: (x[d.loc[x.index, "is_event"]]).sum() / 1000.0),
            reliable_hours=("shortfall_kwh", lambda x: (x <= 0.5).sum()),
            total_hours=("shortfall_kwh", "count"),
        )
        .reset_index()
    )
    per_vehicle["mobility_reliability"] = per_vehicle["reliable_hours"] / per_vehicle["total_hours"]
    return per_vehicle


def _plot_comparison(summary: pd.DataFrame) -> Path:
    strategies = summary.index.tolist()
    revenues = summary["total_revenue_usd"].to_numpy()
    reliability = summary["mobility_reliability"].to_numpy() * 100

    colors = ["#9ca3af", "#60a5fa", "#34d399", "#fbbf24", "#dc2626"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(strategies, revenues, color=colors)
    for i, v in enumerate(revenues):
        ax.text(i, v + (10 if v > 0 else -30), f"${v:,.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Annual revenue per vehicle ($)")
    ax.set_title("Annual revenue by strategy (mean per vehicle)")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    ax.bar(strategies, reliability, color=colors)
    ax.axhline(99, color="red", ls="--", alpha=0.6, label="99% target")
    ax.set_ylabel("Mobility reliability (% hours)")
    ax.set_title("Mobility reliability by strategy")
    ax.set_ylim(min(90, reliability.min() - 1), 100.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out = RESULTS_DIR / "backtest_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def run(n_vehicles: int, horizon_days: int, n_workers: int | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    fleet = pd.read_parquet(DATA_DIR / "fleet.parquet").head(n_vehicles)
    grid = pd.read_parquet(DATA_DIR / "grid_events.parquet")
    grid["timestamp"] = pd.to_datetime(grid["timestamp"])
    lmp = pd.read_parquet(DATA_DIR / "lmp.parquet")
    lmp["timestamp"] = pd.to_datetime(lmp["timestamp"])
    inference = pd.read_parquet(DATA_DIR / "inference_demand.parquet")
    inference["timestamp"] = pd.to_datetime(inference["timestamp"])

    start = pd.Timestamp(SIM_START, tz=PHOENIX_TZ)
    end = start + pd.Timedelta(days=horizon_days)

    print(f"\nbacktest: {n_vehicles} vehicles × {horizon_days} days, {n_workers} workers")
    print(f"  start {start.date()} → end {end.date()}")

    from src.simulator.baselines import STRATEGIES
    metric_rows: list[pd.DataFrame] = []
    for name in STRATEGIES:
        print(f"\n[{name}]")
        decisions = _run_strategy(name, fleet, start, end, n_workers=n_workers)
        decisions.to_parquet(RESULTS_DIR / f"backtest_decisions_{name}.parquet", index=False)
        m = _compute_metrics(decisions, grid, lmp, inference)
        m.insert(0, "strategy", name)
        metric_rows.append(m)

    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv", index=False)

    summary = (
        metrics.groupby("strategy")
        [
            [
                "total_revenue_usd",
                "v2g_revenue_usd",
                "inference_revenue_usd",
                "charging_cost_usd",
                "total_charge_kwh",
                "total_discharge_kwh",
                "infer_hours",
                "mwh_in_events",
                "mobility_reliability",
            ]
        ]
        .mean()
    )
    # Preserve canonical strategy order
    summary = summary.reindex(list(STRATEGIES.keys()))
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv")

    print("\n=== summary (mean per vehicle) ===")
    print(summary.round(2).to_string())

    plot_path = _plot_comparison(summary)
    print(f"\nsummary plot → {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vehicles", type=int, default=20)
    ap.add_argument("--horizon-days", type=int, default=SIM_HORIZON_DAYS)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    run(n_vehicles=args.n_vehicles, horizon_days=args.horizon_days, n_workers=args.workers)


if __name__ == "__main__":
    main()
