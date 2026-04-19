"""Canonical parameter values for the OpenVPP Orchestrator.

All verbatim challenge parameters and derived constants live here so every
downstream module pulls from a single source of truth. See
``docs/assumptions.md`` for rationale and citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


MASTER_SEED: int = 42
SIM_YEAR: int = 2026
SIM_START: date = date(SIM_YEAR, 1, 1)
SIM_HORIZON_DAYS: int = 365
PHOENIX_TZ: str = "America/Phoenix"


@dataclass(frozen=True)
class BatteryParams:
    capacity_kwh: float = 100.0
    reserve_min_kwh: float = 25.0
    reserve_max_kwh: float = 30.0
    flexible_min_kwh: float = 30.0
    flexible_max_kwh: float = 50.0
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    p_charge_max_kw: float = 11.0
    p_discharge_max_kw: float = 11.0


@dataclass(frozen=True)
class DrivingParams:
    daily_kwh_min: float = 25.0
    daily_kwh_max: float = 40.0
    weekday_depart_mean_hour: float = 8.0
    weekday_depart_sd_hour: float = 0.75
    weekday_return_mean_hour: float = 17.5
    weekday_return_sd_hour: float = 1.0
    wfh_probability: float = 0.20
    weekend_trip_probability: float = 0.55
    weekend_trip_kwh_mean: float = 12.0
    weekend_trip_kwh_sd: float = 4.0


@dataclass(frozen=True)
class FleetParams:
    n_vehicles: int = 100
    private_share: float = 0.70
    lambda_miss_private_usd_per_h: float = 50.0
    lambda_miss_fleet_usd_per_h: float = 75.0
    phoenix_lat_min: float = 33.30
    phoenix_lat_max: float = 33.75
    phoenix_lon_min: float = -112.35
    phoenix_lon_max: float = -111.75


@dataclass(frozen=True)
class GridEventParams:
    n_summer: int = 6
    n_winter: int = 4
    summer_months: tuple[int, ...] = (6, 7, 8, 9)
    winter_months: tuple[int, ...] = (12, 1, 2)
    summer_hour_start: int = 16
    summer_hour_end: int = 19
    winter_hour_start: int = 6
    winter_hour_end: int = 9
    duration_h_min: int = 3
    duration_h_max: int = 5


@dataclass(frozen=True)
class LMPParams:
    offpeak_base_usd: float = 0.09
    onpeak_base_usd: float = 0.18
    onpeak_hours: tuple[int, ...] = (15, 16, 17, 18, 19, 20)
    ar1_rho: float = 0.55
    ar1_sigma_usd: float = 0.015
    event_spike_min_usd: float = 0.15
    event_spike_max_usd: float = 0.25
    clamp_min_usd: float = 0.05
    clamp_max_usd: float = 0.60


@dataclass(frozen=True)
class InferenceParams:
    daytime_rate_per_h: float = 0.05
    overnight_rate_per_h: float = 0.01
    daytime_hours: tuple[int, ...] = tuple(range(8, 20))
    session_duration_min_h: int = 4
    session_duration_max_h: int = 10
    power_min_kw: float = 2.0
    power_max_kw: float = 8.0
    revenue_min_usd_per_kwh: float = 0.20
    revenue_max_usd_per_kwh: float = 0.60


@dataclass(frozen=True)
class OptimizerParams:
    horizon_h: int = 24
    n_scenarios: int = 5
    reoptimize_every_h: int = 1


@dataclass(frozen=True)
class Config:
    battery: BatteryParams = field(default_factory=BatteryParams)
    driving: DrivingParams = field(default_factory=DrivingParams)
    fleet: FleetParams = field(default_factory=FleetParams)
    grid: GridEventParams = field(default_factory=GridEventParams)
    lmp: LMPParams = field(default_factory=LMPParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    optimizer: OptimizerParams = field(default_factory=OptimizerParams)


CONFIG = Config()


def seed_for(stream: str) -> int:
    """Derive a per-module seed from MASTER_SEED and a stream label.

    Using a fixed mapping keeps module seeds stable across refactors so long
    as the stream name is unchanged.
    """
    return (MASTER_SEED * 2654435761 + hash(stream)) & 0x7FFFFFFF
