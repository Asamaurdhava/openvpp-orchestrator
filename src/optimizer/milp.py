"""Stochastic mixed-integer linear program for single-vehicle dispatch.

Formulation (per vehicle, per 24h horizon):

**Variables** (t = hour index 0..H-1, s = scenario index 0..S-1):

* ``charge[t,s]`` ∈ [0, P_charge_max] — kW drawn from grid (continuous)
* ``discharge[t,s]`` ∈ [0, P_discharge_max] — kW sent to grid (continuous)
* ``infer[t,s]`` ∈ {0, 1} — inference session active this hour (binary)
* ``shortfall[t,s]`` ≥ 0 — kWh below reserve_min (continuous; soft constraint)
* ``soc[t,s]`` ∈ [0, capacity] — state of charge in kWh at end of hour t

**Non-anticipativity**: ``charge[0,s]``, ``discharge[0,s]``, ``infer[0,s]``
are shared across all scenarios (first-stage here-and-now decision).

**SoC dynamics**:
    soc[t,s] = soc[t-1,s] + η_c·charge[t,s] − (1/η_d)·discharge[t,s]
               − infer_power[t,s]·infer[t,s] − trip_kwh[t]

**Availability gating**: charge, discharge, infer are forced to 0 when the
vehicle is not at home (driving or at work without a V2G-capable charger).

**Inference gating**: infer[t,s] ≤ demand_available[t,s].

**Mobility reliability** (soft): ``soc[t,s] ≥ reserve_min − shortfall[t,s]``.
Shortfall is charged at ``λ / reserve_min`` per kWh·h, so a full-reserve
deficit for one hour costs ``λ``.

**Objective** (maximize expected reward):
    (1/S) · Σ_s Σ_t [
        LMP[t,s]·discharge[t,s]
        − LMP[t,s]·charge[t,s]
        + infer_price[t,s]·infer_power[t,s]·infer[t,s]
        − (λ/reserve_min)·shortfall[t,s]
    ]

The first-stage decision is what the MPC actually executes; downstream
hours are notional futures used to value that decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pulp

from src.config import CONFIG, BatteryParams


@dataclass
class OptimizationResult:
    decisions: pd.DataFrame   # columns: timestamp, charge_kw, discharge_kw, infer, soc_kwh
    first_stage: dict         # {"charge_kw", "discharge_kw", "infer", "timestamp"}
    objective_value: float    # expected reward over scenarios, in USD
    solve_time_s: float
    status: str


def _required_scenario_columns() -> set[str]:
    return {
        "timestamp",
        "is_event",
        "severity",
        "lmp_usd_per_kwh",
        "demand_available",
        "power_kw",
        "revenue_usd_per_kwh",
    }


def _required_availability_columns() -> set[str]:
    return {"timestamp", "is_parked", "is_at_home", "trip_kwh_this_hour"}


def solve_milp(
    soc_init_kwh: float,
    lambda_miss_usd_per_h: float,
    scenarios: Sequence[pd.DataFrame],
    availability_window: pd.DataFrame,
    battery: BatteryParams = CONFIG.battery,
    reserve_kwh: float | None = None,
    cycling_cost_usd_per_kwh: float = 0.002,
    enable_discharge: bool = True,
    enable_inference: bool = True,
    solver_msg: bool = False,
    time_limit_s: float | None = None,
) -> OptimizationResult:
    """Solve the stochastic MILP for a single vehicle over a 24h horizon.

    Parameters
    ----------
    soc_init_kwh : float
        State of charge at the start of hour 0 (before any action).
    lambda_miss_usd_per_h : float
        Mobility failure penalty, in USD per hour of complete reserve
        deficit (0 kWh held against reserve_min for 1 hour). Linear.
    scenarios : list of DataFrame
        Each scenario DataFrame spans exactly H hours and contains the
        columns in :func:`_required_scenario_columns`.
    availability_window : DataFrame
        H rows, scenario-independent, with columns
        (``timestamp``, ``is_parked``, ``is_at_home``, ``trip_kwh_this_hour``).
    reserve_kwh : float, optional
        Reserve threshold for shortfall penalty. Defaults to mid-range
        of ``battery.reserve_min_kwh`` and ``battery.reserve_max_kwh``.
    """
    if reserve_kwh is None:
        reserve_kwh = 0.5 * (battery.reserve_min_kwh + battery.reserve_max_kwh)

    S = len(scenarios)
    if S == 0:
        raise ValueError("Need at least one scenario")

    H = len(scenarios[0])
    for sc in scenarios:
        if len(sc) != H:
            raise ValueError(f"All scenarios must have length {H}; got {len(sc)}")
        missing = _required_scenario_columns() - set(sc.columns)
        if missing:
            raise ValueError(f"Scenario missing columns: {missing}")

    if len(availability_window) != H:
        raise ValueError(f"Availability window must have {H} rows; got {len(availability_window)}")
    missing = _required_availability_columns() - set(availability_window.columns)
    if missing:
        raise ValueError(f"Availability window missing columns: {missing}")

    is_at_home = availability_window["is_at_home"].to_numpy().astype(bool)
    trip_kwh = availability_window["trip_kwh_this_hour"].to_numpy().astype(float)
    timestamps = pd.to_datetime(availability_window["timestamp"])

    # Parameter arrays per scenario
    lmp = np.array([sc["lmp_usd_per_kwh"].to_numpy() for sc in scenarios])
    demand = np.array([sc["demand_available"].to_numpy().astype(bool) for sc in scenarios])
    infer_power = np.array([sc["power_kw"].to_numpy() for sc in scenarios])
    infer_price = np.array([sc["revenue_usd_per_kwh"].to_numpy() for sc in scenarios])

    # ---- LP model ----
    m = pulp.LpProblem("vehicle_dispatch", pulp.LpMaximize)

    charge = [
        [pulp.LpVariable(f"c_{t}_{s}", lowBound=0, upBound=battery.p_charge_max_kw)
         for s in range(S)] for t in range(H)
    ]
    discharge = [
        [pulp.LpVariable(f"d_{t}_{s}", lowBound=0, upBound=battery.p_discharge_max_kw)
         for s in range(S)] for t in range(H)
    ]
    # Inference is binary per scenario.
    infer = [
        [pulp.LpVariable(f"i_{t}_{s}", cat=pulp.LpBinary) for s in range(S)]
        for t in range(H)
    ]
    shortfall = [
        [pulp.LpVariable(f"sh_{t}_{s}", lowBound=0) for s in range(S)]
        for t in range(H)
    ]
    soc = [
        [pulp.LpVariable(f"soc_{t}_{s}", lowBound=0, upBound=battery.capacity_kwh)
         for s in range(S)] for t in range(H)
    ]

    # Non-anticipativity at t=0
    for s in range(1, S):
        m += charge[0][s] == charge[0][0], f"nac_c_{s}"
        m += discharge[0][s] == discharge[0][0], f"nac_d_{s}"
        m += infer[0][s] == infer[0][0], f"nac_i_{s}"

    eta_c = battery.charge_efficiency
    inv_eta_d = 1.0 / battery.discharge_efficiency

    for t in range(H):
        for s in range(S):
            # SoC dynamics
            prev = soc_init_kwh if t == 0 else soc[t - 1][s]
            infer_energy = infer_power[s, t] * infer[t][s]
            m += (
                soc[t][s]
                == prev
                + eta_c * charge[t][s]
                - inv_eta_d * discharge[t][s]
                - infer_energy
                - trip_kwh[t]
            ), f"soc_dyn_{t}_{s}"

            # Availability gating: not at home → no battery-grid exchange or inference
            if not is_at_home[t]:
                m += charge[t][s] == 0, f"nocharge_{t}_{s}"
                m += discharge[t][s] == 0, f"nodischarge_{t}_{s}"
                m += infer[t][s] == 0, f"noinfer_{t}_{s}"

            # Strategy-level disables (for baselines)
            if not enable_discharge:
                m += discharge[t][s] == 0, f"strategy_nodischarge_{t}_{s}"
            if not enable_inference:
                m += infer[t][s] == 0, f"strategy_noinfer_{t}_{s}"

            # Inference must have demand offered
            if not demand[s, t]:
                m += infer[t][s] == 0, f"nodemand_{t}_{s}"

            # Mobility shortfall (soft)
            m += soc[t][s] + shortfall[t][s] >= reserve_kwh, f"reserve_{t}_{s}"

    # ---- Objective: expected reward ----
    lam_per_kwh_h = lambda_miss_usd_per_h / reserve_kwh
    kappa = cycling_cost_usd_per_kwh
    total = 0.0
    for s in range(S):
        reward_s = 0.0
        for t in range(H):
            reward_s += lmp[s, t] * discharge[t][s]
            reward_s -= lmp[s, t] * charge[t][s]
            reward_s += infer_price[s, t] * infer_power[s, t] * infer[t][s]
            reward_s -= lam_per_kwh_h * shortfall[t][s]
            # Cycling / degeneracy-breaking cost
            reward_s -= kappa * (charge[t][s] + discharge[t][s])
        total += reward_s
    m += total / S

    solver_kwargs: dict = {"msg": solver_msg}
    if time_limit_s is not None:
        solver_kwargs["timeLimit"] = time_limit_s
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)

    t0 = time.perf_counter()
    status_code = m.solve(solver)
    solve_time = time.perf_counter() - t0
    status = pulp.LpStatus[status_code]

    # Extract: average trajectory across scenarios (for plotting / backtest)
    charge_mean = np.array([np.mean([pulp.value(charge[t][s]) or 0.0 for s in range(S)]) for t in range(H)])
    discharge_mean = np.array([np.mean([pulp.value(discharge[t][s]) or 0.0 for s in range(S)]) for t in range(H)])
    infer_mean = np.array([np.mean([pulp.value(infer[t][s]) or 0.0 for s in range(S)]) for t in range(H)])
    soc_mean = np.array([np.mean([pulp.value(soc[t][s]) or 0.0 for s in range(S)]) for t in range(H)])

    decisions = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "charge_kw": charge_mean,
            "discharge_kw": discharge_mean,
            "infer": infer_mean,  # in [0,1]; equals the scenario-average activation
            "soc_kwh": soc_mean,
        }
    )

    first_stage = {
        "timestamp": timestamps.iloc[0],
        "charge_kw": float(pulp.value(charge[0][0]) or 0.0),
        "discharge_kw": float(pulp.value(discharge[0][0]) or 0.0),
        "infer": int(round(pulp.value(infer[0][0]) or 0.0)),
    }

    return OptimizationResult(
        decisions=decisions,
        first_stage=first_stage,
        objective_value=float(pulp.value(m.objective) or 0.0),
        solve_time_s=solve_time,
        status=status,
    )
