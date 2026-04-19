"""Rolling-horizon MPC replay for a single vehicle.

Simulates one year of vehicle operation by repeatedly:

1. Sampling ``n_scenarios`` 24h-ahead futures from the forecasters
2. Solving the stochastic MILP
3. Committing the first ``replan_every_h`` hours of the plan to the
   *actual* trace (from ``data/*.parquet``)
4. Updating the SoC
5. Advancing ``replan_every_h`` hours and repeating

Returns a DataFrame with one row per hour: the committed action, the
realized LMP / inference state, and the resulting SoC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.config import CONFIG, PHOENIX_TZ
from src.forecasting.forecasters import sample_scenarios_for_vehicle
from src.optimizer.milp import solve_milp


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ActualTraces:
    """Full-year realized signals. Built once and shared across vehicles."""

    availability: pd.DataFrame        # cols: vehicle_id, timestamp, is_parked, is_at_home
    trips: pd.DataFrame
    grid: pd.DataFrame                # cols: timestamp, is_event, severity
    lmp: pd.DataFrame                 # cols: timestamp, lmp_usd_per_kwh
    inference: pd.DataFrame           # cols: vehicle_id, timestamp, demand_available, power_kw, revenue_usd_per_kwh

    @classmethod
    def load(cls) -> "ActualTraces":
        availability = pd.read_parquet(DATA_DIR / "availability.parquet")
        availability["timestamp"] = pd.to_datetime(availability["timestamp"])
        trips = pd.read_parquet(DATA_DIR / "trips.parquet")
        trips["depart_ts"] = pd.to_datetime(trips["depart_ts"])
        trips["return_ts"] = pd.to_datetime(trips["return_ts"])
        grid = pd.read_parquet(DATA_DIR / "grid_events.parquet")
        grid["timestamp"] = pd.to_datetime(grid["timestamp"])
        lmp = pd.read_parquet(DATA_DIR / "lmp.parquet")
        lmp["timestamp"] = pd.to_datetime(lmp["timestamp"])
        inference = pd.read_parquet(DATA_DIR / "inference_demand.parquet")
        inference["timestamp"] = pd.to_datetime(inference["timestamp"])
        return cls(availability=availability, trips=trips, grid=grid, lmp=lmp, inference=inference)


def _trip_kwh_per_hour(
    trips: pd.DataFrame, vehicle_id: str, hours: pd.DatetimeIndex
) -> np.ndarray:
    """Return kWh subtracted in each hour. Concentrated at trip departure hour."""
    out = np.zeros(len(hours))
    vtrips = trips[trips["vehicle_id"] == vehicle_id]
    if vtrips.empty:
        return out
    hour_to_idx = {ts: i for i, ts in enumerate(hours)}
    for _, trip in vtrips.iterrows():
        dep_hour = trip["depart_ts"].floor("h")
        if dep_hour in hour_to_idx:
            out[hour_to_idx[dep_hour]] += float(trip["kwh"])
    return out


def _inject_actual_at_t0(
    scenarios: list[pd.DataFrame],
    actual_lmp: float,
    actual_is_event: bool,
    actual_severity: float,
    actual_demand: bool,
    actual_inf_power: float,
    actual_inf_price: float,
) -> list[pd.DataFrame]:
    """Set t=0 of every scenario to observed values.

    Ensures the first-stage decision is conditional on reality rather than
    a (possibly wrong) sampled t=0. Non-anticipativity then propagates this
    to all scenarios automatically.
    """
    for sc in scenarios:
        sc.loc[sc.index[0], "lmp_usd_per_kwh"] = actual_lmp
        sc.loc[sc.index[0], "is_event"] = actual_is_event
        sc.loc[sc.index[0], "severity"] = actual_severity
        sc.loc[sc.index[0], "demand_available"] = actual_demand
        sc.loc[sc.index[0], "power_kw"] = actual_inf_power
        sc.loc[sc.index[0], "revenue_usd_per_kwh"] = actual_inf_price
    return scenarios


def simulate_vehicle_mpc(
    vehicle: pd.Series,
    traces: ActualTraces,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_h: int = 24,
    replan_every_h: int = 1,
    n_scenarios: int = 2,
    soc_init_kwh: float | None = None,
    seed: int = 2026,
    enable_discharge: bool = True,
    enable_inference: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Simulate one vehicle with MPC rolling horizon.

    The ``enable_discharge`` and ``enable_inference`` flags gate which
    actions the optimizer may choose. They are the single knob by which
    we derive smart-charge-only and V2G-only baselines from the same core.
    """
    vehicle_id = vehicle["vehicle_id"]
    lam = float(vehicle["lambda_miss_usd_per_h"])

    # Full-year hour index for this vehicle
    hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz=PHOENIX_TZ)
    n_hours = len(hours)

    v_avail = traces.availability[traces.availability["vehicle_id"] == vehicle_id].sort_values("timestamp")
    v_avail_map = v_avail.set_index("timestamp")
    v_inf = traces.inference[traces.inference["vehicle_id"] == vehicle_id].sort_values("timestamp")
    v_inf_map = v_inf.set_index("timestamp")
    lmp_map = traces.lmp.set_index("timestamp")["lmp_usd_per_kwh"]
    grid_map = traces.grid.set_index("timestamp")

    trip_kwh = _trip_kwh_per_hour(traces.trips, vehicle_id, hours)

    if soc_init_kwh is None:
        soc_init_kwh = CONFIG.battery.reserve_max_kwh + 0.5 * CONFIG.battery.flexible_max_kwh

    # Execution buffers
    charge_exec = np.zeros(n_hours)
    discharge_exec = np.zeros(n_hours)
    infer_exec = np.zeros(n_hours, dtype=int)
    soc_exec = np.zeros(n_hours)
    revenue_exec = np.zeros(n_hours)
    shortfall_exec = np.zeros(n_hours)

    eta_c = CONFIG.battery.charge_efficiency
    inv_eta_d = 1.0 / CONFIG.battery.discharge_efficiency

    soc = soc_init_kwh
    t = 0
    n_solves = 0
    total_solve_s = 0.0

    while t < n_hours:
        asof_ts = hours[t]
        plan_len = min(horizon_h, n_hours - t)

        # Build availability window
        window_hours = hours[t : t + plan_len]
        window_avail = v_avail_map.reindex(window_hours)
        window_trip_kwh = trip_kwh[t : t + plan_len]
        availability_window = pd.DataFrame(
            {
                "timestamp": window_hours,
                "is_parked": window_avail["is_parked"].to_numpy(),
                "is_at_home": window_avail["is_at_home"].to_numpy(),
                "trip_kwh_this_hour": window_trip_kwh,
            }
        )

        scenarios = sample_scenarios_for_vehicle(
            vehicle_id=vehicle_id,
            asof_ts=asof_ts,
            horizon_h=plan_len,
            n_scenarios=n_scenarios,
            seed=seed,
        )

        # Inject observed t=0 so first-stage sees reality.
        actual_lmp = float(lmp_map.get(asof_ts, np.nan))
        grid_row = grid_map.loc[asof_ts] if asof_ts in grid_map.index else None
        actual_is_event = bool(grid_row["is_event"]) if grid_row is not None else False
        actual_severity = float(grid_row["severity"]) if grid_row is not None else 0.0
        inf_row_now = v_inf_map.loc[asof_ts] if asof_ts in v_inf_map.index else None
        actual_demand_now = bool(inf_row_now["demand_available"]) if inf_row_now is not None else False
        actual_inf_power_now = float(inf_row_now["power_kw"]) if inf_row_now is not None else 0.0
        actual_inf_price_now = float(inf_row_now["revenue_usd_per_kwh"]) if inf_row_now is not None else 0.0
        scenarios = _inject_actual_at_t0(
            scenarios,
            actual_lmp,
            actual_is_event,
            actual_severity,
            actual_demand_now,
            actual_inf_power_now,
            actual_inf_price_now,
        )

        result = solve_milp(
            soc_init_kwh=soc,
            lambda_miss_usd_per_h=lam,
            scenarios=scenarios,
            availability_window=availability_window,
            enable_discharge=enable_discharge,
            enable_inference=enable_inference,
        )
        n_solves += 1
        total_solve_s += result.solve_time_s

        # Commit first `replan_every_h` hours of the plan to the actual trace.
        commit_n = min(replan_every_h, plan_len)
        decisions = result.decisions.iloc[:commit_n]

        for j in range(commit_n):
            idx = t + j
            ts = hours[idx]
            is_home = bool(v_avail_map.at[ts, "is_at_home"])
            actual_lmp = float(lmp_map.get(ts, np.nan))
            inf_row = v_inf_map.loc[ts] if ts in v_inf_map.index else None
            actual_demand = bool(inf_row["demand_available"]) if inf_row is not None else False
            actual_inf_power = float(inf_row["power_kw"]) if inf_row is not None else 0.0
            actual_inf_price = float(inf_row["revenue_usd_per_kwh"]) if inf_row is not None else 0.0

            c = 0.0 if not is_home else float(decisions["charge_kw"].iloc[j])
            d = 0.0 if not is_home else float(decisions["discharge_kw"].iloc[j])
            # Planned infer is scenario-averaged; threshold at 0.5 for execution.
            want_infer = (
                is_home and actual_demand and float(decisions["infer"].iloc[j]) >= 0.5
            )
            infer_active = int(want_infer)
            infer_energy = actual_inf_power if infer_active else 0.0

            # Apply to SoC
            new_soc = soc + eta_c * c - inv_eta_d * d - infer_energy - trip_kwh[idx]
            # Enforce battery physical bounds; if we're forced to clip, this is a model
            # inconsistency and we log shortfall appropriately.
            if new_soc < 0:
                # Can't discharge or run inference past empty. Clip and log shortfall.
                overshoot = -new_soc
                if infer_active and infer_energy >= overshoot:
                    infer_active = 0
                    infer_energy = 0.0
                    new_soc = soc + eta_c * c - inv_eta_d * d - trip_kwh[idx]
                if new_soc < 0:
                    d = max(0.0, d - (-new_soc) * CONFIG.battery.discharge_efficiency)
                    new_soc = soc + eta_c * c - inv_eta_d * d - infer_energy - trip_kwh[idx]
                new_soc = max(new_soc, 0.0)
            new_soc = min(new_soc, CONFIG.battery.capacity_kwh)

            reserve_kwh = 0.5 * (CONFIG.battery.reserve_min_kwh + CONFIG.battery.reserve_max_kwh)
            shortfall = max(0.0, reserve_kwh - new_soc)

            # Revenue accounting (actual prices, not forecast)
            rev = (
                actual_lmp * d
                - actual_lmp * c
                + (actual_inf_price * actual_inf_power if infer_active else 0.0)
                - (lam / reserve_kwh) * shortfall
            )

            charge_exec[idx] = c
            discharge_exec[idx] = d
            infer_exec[idx] = infer_active
            soc_exec[idx] = new_soc
            revenue_exec[idx] = rev
            shortfall_exec[idx] = shortfall

            soc = new_soc

        t += commit_n

    if verbose:
        print(
            f"  {vehicle_id}: {n_solves} solves, mean {total_solve_s/max(n_solves,1)*1000:.1f}ms/solve"
        )

    return pd.DataFrame(
        {
            "timestamp": hours,
            "vehicle_id": vehicle_id,
            "charge_kw": charge_exec,
            "discharge_kw": discharge_exec,
            "infer_active": infer_exec,
            "soc_kwh": soc_exec,
            "revenue_usd": revenue_exec,
            "shortfall_kwh": shortfall_exec,
        }
    )


def simulate_vehicle_stochastic_coopt(*args, **kwargs) -> pd.DataFrame:
    """Full co-optimization: charge + discharge + inference."""
    return simulate_vehicle_mpc(*args, enable_discharge=True, enable_inference=True, **kwargs)


def simulate_vehicle_smart_charge(*args, **kwargs) -> pd.DataFrame:
    """Smart-charge only (no V2G, no inference) — optimizes charging cost."""
    return simulate_vehicle_mpc(*args, enable_discharge=False, enable_inference=False, **kwargs)


def simulate_vehicle_v2g_only(*args, **kwargs) -> pd.DataFrame:
    """V2G only (charge + discharge, no inference)."""
    return simulate_vehicle_mpc(*args, enable_discharge=True, enable_inference=False, **kwargs)
