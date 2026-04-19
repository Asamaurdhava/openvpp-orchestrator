# Architecture

OpenVPP Orchestrator is a pipeline: synthetic signals → forecasters → stochastic MILP → per-vehicle dispatch → backtest + dashboard.

## Data flow

```
                 ┌─────────────────────────┐
                 │   SYNTHETIC FLEET DATA  │
                 │  100 EVs, 1-year traces │
                 └───────────┬─────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
 ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
 │ GRID EVENTS  │   │  LMP PRICES   │   │ INFERENCE    │
 │  forecaster  │   │   forecaster  │   │   DEMAND     │
 └──────┬───────┘   └───────┬───────┘   └──────┬───────┘
        │                   │                   │
        └───────────┬───────┴───────────┬───────┘
                    ▼                   ▼
              ┌──────────────────────────────┐
              │  ROLLING-HORIZON STOCHASTIC  │
              │         MILP OPTIMIZER       │
              │  (mobility-risk priced)      │
              └───────────┬──────────────────┘
                          │
                          ▼
              ┌──────────────────────────────┐
              │  DECISIONS PER VEHICLE/HOUR  │
              │ {charge, V2G, infer, idle}   │
              └───────────┬──────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 ┌─────────────────┐          ┌─────────────────────┐
 │  BACKTEST +     │          │  STREAMLIT          │
 │  5 BASELINES    │          │  DASHBOARD          │
 └─────────────────┘          └─────────────────────┘
```

## Component contracts

Every module below exposes pure functions with explicit dataframe-in / dataframe-out contracts. No hidden state, no globals. This keeps unit testing trivial and enables per-vehicle parallel solves.

### `src/fleet/generator.py`

**Input:** `seed: int`, `n_vehicles: int`, `start_date: date`, `horizon_days: int`
**Output:** DataFrame (`vehicle_id`, `ownership`, `home_lat`, `home_lon`, `work_lat`, `work_lon`, `commute_weekday_depart`, `commute_weekday_return`, `trip_kwh_mean`, `lambda_miss_usd`)
**Side effects:** writes `data/fleet.parquet`.

### `src/signals/grid_events.py`

**Input:** `seed`, `year: int`, n_summer=6, n_winter=4
**Output:** DataFrame (`timestamp`, `is_event: bool`, `event_id: int|None`, `severity: float`)
**Contract:** exactly `n_summer + n_winter` events, durations 3–5h drawn uniformly, summer in Jun–Sep afternoons, winter in Dec–Feb mornings.

### `src/signals/lmp.py`

**Input:** `seed`, `grid_events_df`
**Output:** DataFrame (`timestamp`, `lmp_usd_per_kwh`)
**Contract:** diurnal baseline + AR(1) noise + additive spike during event hours. Range clamped to [0.05, 0.60].

### `src/signals/inference_demand.py`

**Input:** `seed`, `vehicle_ids`, `start_date`, `horizon_days`
**Output:** DataFrame (`vehicle_id`, `timestamp`, `inference_active: bool`, `power_kw: float`, `revenue_usd_per_kwh: float`)
**Contract:** Poisson-process session starts with time-varying rate; session duration 4–10h; `revenue_usd_per_kwh` sampled per-session from [0.20, 0.60].

### `src/forecasting/forecasters.py`

Three callables:

- `forecast_grid_events(history_df, asof_ts, horizon_h=24, n_scenarios=5) → list[DataFrame]`
- `forecast_lmp(history_df, asof_ts, horizon_h=24, n_scenarios=5) → list[DataFrame]`
- `forecast_inference_demand(history_df, asof_ts, horizon_h=24, n_scenarios=5) → list[DataFrame]`

**Contract:** each returns `n_scenarios` equally-weighted DataFrames representing sampled futures. No single-point forecasts — the optimizer is stochastic by design.

### `src/optimizer/milp.py`

Core PuLP formulation per vehicle. See ACTION_PLAN.md §8 for the formal spec.

**Input:** `vehicle_row: Series`, `soc_init: float`, `scenarios: list[ScenarioBundle]`, `horizon_h: int`, `params: OptimizerParams`
**Output:** DataFrame (`timestamp`, `action ∈ {charge,discharge,infer,idle}`, `power_kw`, `soc_end_kwh`, `expected_reward_usd`)
**Contract:** deterministic given same inputs (seed-fixed). Solve time target < 1s for single-vehicle 24h × 5-scenario instance.

### `src/optimizer/rolling_horizon.py`

**Input:** full 1-year traces + fleet + optimizer params
**Output:** DataFrame (`vehicle_id`, `timestamp`, `action`, `power_kw`, `soc_end_kwh`, `revenue_usd`)
**Contract:** for each hour, runs MILP with the current SoC and re-sampled 24h forecasts; executes only the first hour's decision; advances state; repeats. Vehicles solved in parallel (multiprocessing.Pool).

### `src/simulator/backtest.py`

**Input:** full 1-year decisions, actual traces
**Output:** `results/<strategy>_metrics.csv` with columns (`vehicle_id`, `total_revenue_usd`, `v2g_revenue_usd`, `inference_revenue_usd`, `charging_cost_usd`, `mobility_failures`, `mwh_delivered_in_events`)

### `src/simulator/baselines.py`

Implements five strategies exposing `decide(vehicle_row, state, signals_window) → ActionDecision`:
1. `passive` — charge whenever plugged in, no V2G, no inference
2. `smart_charge` — charge at LMP minima only
3. `v2g_only` — smart charge + V2G during events
4. `greedy` — highest-$ action each hour, deterministic forecasts
5. `stochastic_coopt` — our full optimizer (the defending champion)

### `src/dashboard/app.py`

Streamlit multi-page app. 4 tabs:
1. **Live Fleet View** — pydeck map, 100 markers, action color-coded, earnings ticker
2. **Grid Event Simulator** — one button; triggers a synthetic event and replays the dispatch flip
3. **Economics Sensitivity** — slider $0–$0.60/kWh-eq inference revenue; annual NPV curve
4. **Baseline Comparison** — bar chart across 5 strategies on revenue + reliability

Reads pre-computed results from `results/` — **no live MILP solves during demo**. Event simulator uses cached scenarios.

## Parallelism model

Per-vehicle solves are **embarrassingly parallel**. The rolling-horizon wrapper uses `multiprocessing.Pool(n_jobs=os.cpu_count()-1)`. Inter-vehicle coordination (e.g., aggregate feeder caps) is **not** modeled in v1 — each vehicle optimizes independently. This is noted in README limitations.

## File outputs (all under `data/` and `results/`)

| Path | Producer | Consumer |
|---|---|---|
| `data/fleet.parquet` | fleet/generator | all downstream |
| `data/grid_events_<year>.parquet` | signals/grid_events | lmp, forecasters, backtest |
| `data/lmp_<year>.parquet` | signals/lmp | forecasters, backtest |
| `data/inference_<year>.parquet` | signals/inference_demand | forecasters, backtest |
| `results/<strategy>_decisions.parquet` | rolling_horizon / baselines | backtest, dashboard |
| `results/<strategy>_metrics.csv` | backtest | dashboard, deck |

## Determinism

Every stochastic component takes a `seed`. The top-level entry point (`src/simulator/backtest.py`) fixes a master seed and derives module seeds from it. A clean run from a fixed seed must reproduce bit-identical results.
