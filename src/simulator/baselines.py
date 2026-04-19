"""Five baseline strategies for the backtest.

All five expose the same signature:

    simulate(vehicle, traces, start, end, soc_init_kwh) -> pd.DataFrame

The returned DataFrame has the columns
(timestamp, vehicle_id, charge_kw, discharge_kw, infer_active,
soc_kwh, revenue_usd, shortfall_kwh) — identical to
:func:`src.optimizer.rolling_horizon.simulate_vehicle_mpc`.

Strategies:

1. ``passive`` — charge whenever plugged in. No V2G, no inference.
2. ``smart_charge`` — optimize charging cost (MPC, no V2G, no inference).
3. ``v2g_only`` — MPC with V2G enabled, no inference.
4. ``greedy`` — myopic 1-hour argmax over instantaneous rewards.
5. ``stochastic_coopt`` — our full MILP (Phase 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, PHOENIX_TZ
from src.optimizer.rolling_horizon import (
    ActualTraces,
    _trip_kwh_per_hour,
    simulate_vehicle_smart_charge,
    simulate_vehicle_stochastic_coopt,
    simulate_vehicle_v2g_only,
)


def _empty_output(vehicle_id: str, hours: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(hours)
    return pd.DataFrame(
        {
            "timestamp": hours,
            "vehicle_id": vehicle_id,
            "charge_kw": np.zeros(n),
            "discharge_kw": np.zeros(n),
            "infer_active": np.zeros(n, dtype=int),
            "soc_kwh": np.zeros(n),
            "revenue_usd": np.zeros(n),
            "shortfall_kwh": np.zeros(n),
        }
    )


def simulate_vehicle_passive(
    vehicle: pd.Series,
    traces: ActualTraces,
    start: pd.Timestamp,
    end: pd.Timestamp,
    soc_init_kwh: float | None = None,
    **_,
) -> pd.DataFrame:
    """Charge whenever plugged in at max power. No V2G, no inference."""
    vehicle_id = vehicle["vehicle_id"]
    lam = float(vehicle["lambda_miss_usd_per_h"])
    hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz=PHOENIX_TZ)
    n = len(hours)

    v_avail = traces.availability[traces.availability["vehicle_id"] == vehicle_id].set_index("timestamp")
    is_at_home = v_avail.reindex(hours)["is_at_home"].to_numpy().astype(bool)
    lmp = traces.lmp.set_index("timestamp")["lmp_usd_per_kwh"].reindex(hours).to_numpy()
    trip_kwh = _trip_kwh_per_hour(traces.trips, vehicle_id, hours)

    batt = CONFIG.battery
    eta_c = batt.charge_efficiency
    reserve = 0.5 * (batt.reserve_min_kwh + batt.reserve_max_kwh)

    if soc_init_kwh is None:
        soc_init_kwh = batt.reserve_max_kwh + 0.5 * batt.flexible_max_kwh

    out = _empty_output(vehicle_id, hours)
    soc = soc_init_kwh
    for t in range(n):
        c = 0.0
        if is_at_home[t] and soc < batt.capacity_kwh - 0.1:
            c = min(batt.p_charge_max_kw, (batt.capacity_kwh - soc) / eta_c)
        new_soc = soc + eta_c * c - trip_kwh[t]
        new_soc = max(0.0, min(batt.capacity_kwh, new_soc))
        shortfall = max(0.0, reserve - new_soc)
        revenue = -lmp[t] * c - (lam / reserve) * shortfall

        out.at[t, "charge_kw"] = c
        out.at[t, "soc_kwh"] = new_soc
        out.at[t, "revenue_usd"] = revenue
        out.at[t, "shortfall_kwh"] = shortfall
        soc = new_soc
    return out


def simulate_vehicle_greedy(
    vehicle: pd.Series,
    traces: ActualTraces,
    start: pd.Timestamp,
    end: pd.Timestamp,
    soc_init_kwh: float | None = None,
    **_,
) -> pd.DataFrame:
    """Myopic greedy: each hour, pick the single most profitable action given
    current state, using only **currently observed** LMP/demand (no forecasts).

    Heuristic rules:
    * If demand available and at home and SoC headroom covers session energy, infer.
    * Else if LMP ≥ threshold_high and at home and SoC ≥ reserve + headroom, discharge.
    * Else if LMP ≤ threshold_low and at home and SoC < capacity, charge.
    * Else idle.
    """
    vehicle_id = vehicle["vehicle_id"]
    lam = float(vehicle["lambda_miss_usd_per_h"])
    hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz=PHOENIX_TZ)
    n = len(hours)

    v_avail = traces.availability[traces.availability["vehicle_id"] == vehicle_id].set_index("timestamp")
    is_at_home = v_avail.reindex(hours)["is_at_home"].to_numpy().astype(bool)
    lmp = traces.lmp.set_index("timestamp")["lmp_usd_per_kwh"].reindex(hours).to_numpy()

    v_inf = traces.inference[traces.inference["vehicle_id"] == vehicle_id].set_index("timestamp")
    inf = v_inf.reindex(hours)
    demand = inf["demand_available"].to_numpy().astype(bool)
    inf_power = inf["power_kw"].to_numpy()
    inf_price = inf["revenue_usd_per_kwh"].to_numpy()

    trip_kwh = _trip_kwh_per_hour(traces.trips, vehicle_id, hours)

    batt = CONFIG.battery
    eta_c = batt.charge_efficiency
    inv_eta_d = 1.0 / batt.discharge_efficiency
    reserve = 0.5 * (batt.reserve_min_kwh + batt.reserve_max_kwh)

    lmp_low = 0.11  # charge below this (off-peak)
    lmp_high = 0.15  # discharge above this
    infer_headroom = 10.0  # kWh of buffer above reserve to accept a session hour

    if soc_init_kwh is None:
        soc_init_kwh = batt.reserve_max_kwh + 0.5 * batt.flexible_max_kwh

    out = _empty_output(vehicle_id, hours)
    soc = soc_init_kwh
    for t in range(n):
        c = 0.0
        d = 0.0
        ia = 0
        if is_at_home[t]:
            want_infer = (
                demand[t]
                and inf_price[t] * inf_power[t] > lmp[t] * inf_power[t]
                and soc - inf_power[t] - trip_kwh[t] >= reserve + infer_headroom
            )
            if want_infer:
                ia = 1
            elif lmp[t] >= lmp_high and soc - inv_eta_d * batt.p_discharge_max_kw - trip_kwh[t] >= reserve:
                d = batt.p_discharge_max_kw
            elif lmp[t] <= lmp_low and soc < batt.capacity_kwh:
                c = min(batt.p_charge_max_kw, (batt.capacity_kwh - soc) / eta_c)

        infer_energy = inf_power[t] if ia else 0.0
        new_soc = soc + eta_c * c - inv_eta_d * d - infer_energy - trip_kwh[t]
        if new_soc < 0:
            # Scale back in priority: drop inference, then discharge.
            if ia and infer_energy >= -new_soc:
                ia = 0
                infer_energy = 0.0
            new_soc = soc + eta_c * c - inv_eta_d * d - infer_energy - trip_kwh[t]
            if new_soc < 0 and d > 0:
                d = max(0.0, d + new_soc * batt.discharge_efficiency)
                new_soc = soc + eta_c * c - inv_eta_d * d - infer_energy - trip_kwh[t]
            new_soc = max(new_soc, 0.0)
        new_soc = min(batt.capacity_kwh, new_soc)

        shortfall = max(0.0, reserve - new_soc)
        revenue = (
            lmp[t] * d
            - lmp[t] * c
            + (inf_price[t] * inf_power[t] if ia else 0.0)
            - (lam / reserve) * shortfall
        )

        out.at[t, "charge_kw"] = c
        out.at[t, "discharge_kw"] = d
        out.at[t, "infer_active"] = ia
        out.at[t, "soc_kwh"] = new_soc
        out.at[t, "revenue_usd"] = revenue
        out.at[t, "shortfall_kwh"] = shortfall
        soc = new_soc
    return out


# Strategy registry for easy iteration
STRATEGIES: dict[str, object] = {
    "passive": simulate_vehicle_passive,
    "smart_charge": simulate_vehicle_smart_charge,
    "v2g_only": simulate_vehicle_v2g_only,
    "greedy": simulate_vehicle_greedy,
    "stochastic_coopt": simulate_vehicle_stochastic_coopt,
}
