"""Probabilistic 24h-ahead forecasters for the MILP optimizer.

Three independent samplers + one joint scenario builder:

* :func:`sample_grid_events` — per-hour Bernoulli on a seasonal rate.
* :func:`sample_lmp` — diurnal baseline + AR(1) noise + event-conditional spike.
* :func:`sample_inference_demand` — per-vehicle Poisson session-starts with
  session-level power and price draws.
* :func:`sample_scenarios_for_vehicle` — produces a list of ``n_scenarios``
  DataFrames, each with every column the MILP consumes, jointly coherent
  (LMP spikes align with that scenario's grid events).

All samplers are pure functions of an ``asof_ts``, a horizon length, and a
seed. They can be called from any point in a rolling-horizon replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG, PHOENIX_TZ


@dataclass(frozen=True)
class ForecastSpec:
    """Minimal bundle of forecaster rates so tests/experiments can tweak them.

    Rates calibrated so that integrating across a year yields ~6 summer
    and ~4 winter event *starts*, matching challenge assumptions.
    """

    summer_event_rate_per_h: float = 0.0125   # 6 starts / (4 months × 30 d × 4 h) ≈ 0.0125
    winter_event_rate_per_h: float = 0.0110   # 4 starts / (3 months × 30 d × 4 h) ≈ 0.0111
    offseason_event_rate_per_h: float = 0.0001
    event_avg_duration_h: float = 4.0


DEFAULT_SPEC = ForecastSpec()


def _season_rates(timestamps: pd.DatetimeIndex, spec: ForecastSpec) -> np.ndarray:
    """Per-hour event-start rate conditional on season + time-of-day window."""
    gp = CONFIG.grid
    month = timestamps.month.to_numpy()
    hour = timestamps.hour.to_numpy()

    in_summer_month = np.isin(month, gp.summer_months)
    in_summer_hours = (hour >= gp.summer_hour_start) & (hour <= gp.summer_hour_end)
    in_winter_month = np.isin(month, gp.winter_months)
    in_winter_hours = (hour >= gp.winter_hour_start) & (hour <= gp.winter_hour_end)

    rates = np.full(len(timestamps), spec.offseason_event_rate_per_h, dtype=float)
    rates[in_summer_month & in_summer_hours] = spec.summer_event_rate_per_h
    rates[in_winter_month & in_winter_hours] = spec.winter_event_rate_per_h
    return rates


def sample_grid_events(
    asof_ts: pd.Timestamp,
    horizon_h: int,
    rng: np.random.Generator,
    spec: ForecastSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """Sample a single grid-event scenario over the next ``horizon_h`` hours.

    Returns a DataFrame with columns (timestamp, is_event, severity).
    """
    timestamps = pd.date_range(asof_ts, periods=horizon_h, freq="h")
    rates = _season_rates(timestamps, spec)

    # For each hour, probability of a new event *starting* this hour.
    # Once an event starts, it persists for a duration sampled from [3,5] h.
    is_event = np.zeros(horizon_h, dtype=bool)
    severity = np.zeros(horizon_h, dtype=float)
    gp = CONFIG.grid

    i = 0
    while i < horizon_h:
        if rng.random() < rates[i]:
            dur = int(rng.integers(gp.duration_h_min, gp.duration_h_max + 1))
            sev = float(rng.uniform(0.5, 1.0))
            end = min(horizon_h, i + dur)
            is_event[i:end] = True
            severity[i:end] = sev
            i = end
        else:
            i += 1

    return pd.DataFrame({"timestamp": timestamps, "is_event": is_event, "severity": severity})


def sample_lmp(
    asof_ts: pd.Timestamp,
    horizon_h: int,
    grid_scenario: pd.DataFrame,
    rng: np.random.Generator,
    prev_noise: float = 0.0,
) -> pd.DataFrame:
    """Sample LMP for the horizon conditional on a grid scenario.

    ``prev_noise`` is the AR(1) noise state at the end of history; the
    forecaster continues the chain to avoid a discontinuity at ``asof_ts``.
    """
    lp = CONFIG.lmp
    timestamps = pd.date_range(asof_ts, periods=horizon_h, freq="h")
    hours = timestamps.hour.to_numpy()

    base = np.where(np.isin(hours, lp.onpeak_hours), lp.onpeak_base_usd, lp.offpeak_base_usd)

    noise = np.zeros(horizon_h)
    prev = prev_noise
    for t in range(horizon_h):
        eps = rng.normal(0.0, lp.ar1_sigma_usd)
        noise[t] = lp.ar1_rho * prev + eps
        prev = noise[t]

    spike_band = lp.event_spike_max_usd - lp.event_spike_min_usd
    spike = np.where(
        grid_scenario["is_event"].to_numpy(),
        lp.event_spike_min_usd + grid_scenario["severity"].to_numpy() * spike_band,
        0.0,
    )

    lmp = np.clip(base + noise + spike, lp.clamp_min_usd, lp.clamp_max_usd)
    return pd.DataFrame({"timestamp": timestamps, "lmp_usd_per_kwh": lmp})


def sample_inference_demand(
    asof_ts: pd.Timestamp,
    horizon_h: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample one inference-demand realization for a single vehicle.

    Returns a DataFrame with columns (timestamp, demand_available, power_kw,
    revenue_usd_per_kwh).
    """
    ip = CONFIG.inference
    timestamps = pd.date_range(asof_ts, periods=horizon_h, freq="h")
    hours = timestamps.hour.to_numpy()
    rates = np.where(np.isin(hours, ip.daytime_hours), ip.daytime_rate_per_h, ip.overnight_rate_per_h)

    active = np.zeros(horizon_h, dtype=bool)
    power = np.zeros(horizon_h)
    price = np.zeros(horizon_h)

    i = 0
    while i < horizon_h:
        if rng.random() < rates[i]:
            dur = int(rng.integers(ip.session_duration_min_h, ip.session_duration_max_h + 1))
            p_kw = float(rng.uniform(ip.power_min_kw, ip.power_max_kw))
            p_usd = float(rng.uniform(ip.revenue_min_usd_per_kwh, ip.revenue_max_usd_per_kwh))
            end = min(horizon_h, i + dur)
            active[i:end] = True
            power[i:end] = p_kw
            price[i:end] = p_usd
            i = end
        else:
            i += 1

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_available": active,
            "power_kw": power,
            "revenue_usd_per_kwh": price,
        }
    )


def sample_scenarios_for_vehicle(
    vehicle_id: str,
    asof_ts: pd.Timestamp,
    horizon_h: int = 24,
    n_scenarios: int = 5,
    seed: int = 0,
    spec: ForecastSpec = DEFAULT_SPEC,
) -> list[pd.DataFrame]:
    """Joint scenario samples for one vehicle's 24h-ahead optimization.

    Each returned DataFrame has columns:
    timestamp, is_event, severity, lmp_usd_per_kwh, demand_available,
    power_kw, revenue_usd_per_kwh.

    Scenarios are independent draws; equally-weighted from the optimizer's
    perspective.
    """
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize(PHOENIX_TZ)

    import hashlib
    asof_epoch = int(asof_ts.timestamp())
    vid_hash = int.from_bytes(hashlib.sha1(vehicle_id.encode()).digest()[:4], "big")
    scenarios: list[pd.DataFrame] = []
    for k in range(n_scenarios):
        s = (seed + k * 7919 + vid_hash + asof_epoch * 31) & 0x7FFFFFFF
        rng = np.random.default_rng(s)

        grid = sample_grid_events(asof_ts, horizon_h, rng, spec)
        lmp = sample_lmp(asof_ts, horizon_h, grid, rng)
        inf = sample_inference_demand(asof_ts, horizon_h, rng)

        merged = grid.merge(lmp, on="timestamp").merge(inf, on="timestamp")
        scenarios.append(merged)

    return scenarios
