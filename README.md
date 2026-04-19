# OpenVPP Orchestrator

**AI orchestration for EV batteries across mobility, grid services, and distributed inference.**

Submission for the OpenVPP Challenge — *Software For Energy*, Arizona track.

---

## The problem

Every electric vehicle carries a 100 kWh battery that sits unused 18–22 hours a day. That same battery can do three different jobs: **drive the owner**, **feed the grid during stress events**, or **run AI inference workloads**. These jobs compete for a limited 30–50 kWh flexible capacity per vehicle. No shipping platform today co-optimizes all three under real-world uncertainty.

We build that orchestrator.

## Our approach — three design choices

1. **Mobility risk is priced, not walled off.** Instead of a hard minimum state-of-charge constraint, we attach a dollar penalty (λ ≈ $50) to mobility shortfall. The optimizer weighs the cost of a missed trip against grid / inference revenue. Outcome: more flexible participation with ≥99% mobility reliability.

2. **Three-way co-optimization.** V2G alone is worth ~$20–150/year per vehicle (rare events). Inference alone is worth ~$400–2,000/year (always-on demand). We stack them, and when one disappears the system auto-routes to the other. The system degrades gracefully.

3. **Economic honesty built-in.** The dashboard ships with a sensitivity slider for inference revenue. Set it to $0 and the system becomes a pure V2G + smart-charging platform — still valuable, still working. Protects the design against "what if the EV-inference market doesn't materialize?"

## Technical core

Rolling-horizon stochastic mixed-integer linear program (MILP), solved per-vehicle in parallel. Three probabilistic forecasters feed scenario samples for 24-hour ahead planning; the decision from the first hour is executed, the state advances, and the problem re-solves every hour. See `docs/architecture.md` for the full component diagram and `docs/assumptions.md` for the parameter sheet.

## Setup

Requires Python 3.12 (3.11 also works). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import pulp, pandas, numpy, plotly, streamlit, pydeck, pptx; print('ok')"
```

## Reproduce the results

The shortest path, using the Makefile:

```bash
make install     # create venv + install deps  (~2 min)
make reproduce   # generate data + backtest + deck  (~30 min)
make dash        # launch Streamlit dashboard
```

The explicit pipeline (equivalent, easier to run piecewise):

```bash
# 1. Generate synthetic fleet + 1-year signal traces
python -m src.fleet.generator
python -m src.signals.grid_events
python -m src.signals.lmp
python -m src.signals.inference_demand
python -m src.sanity_check          # verifies all traces match challenge ranges

# 2. Quick visual sanity of the forecasters (optional)
python -m src.forecasting.demo

# 3. Single-vehicle MILP demo (optional, ~10 s)
python -m src.optimizer.demo

# 4. Full backtest across 5 strategies
python -m src.simulator.backtest --n-vehicles 10 --horizon-days 365

# 5. Build the pitch deck from the backtest summary
python -m src.pitch.deck

# 6. Launch the dashboard
streamlit run src/dashboard/app.py
```

### Strategies compared in the backtest

| # | Strategy | What it does |
|---|---|---|
| 1 | `passive` | Charge whenever plugged in. No V2G, no inference. |
| 2 | `smart_charge` | MILP with V2G + inference disabled — cost-minimizing charging only. |
| 3 | `v2g_only` | MILP with inference disabled — classic V2G platform. |
| 4 | `greedy` | Myopic 1-hour argmax, no forecasting, no planning. |
| 5 | `stochastic_coopt` | **Ours** — full stochastic MILP with 3-way co-optimization. |

All five use the same underlying dispatch execution engine so revenue numbers are strictly comparable. See [src/simulator/baselines.py](src/simulator/baselines.py) and [src/optimizer/rolling_horizon.py](src/optimizer/rolling_horizon.py).

## Assumptions

All challenge parameters are captured verbatim in [`docs/assumptions.md`](docs/assumptions.md), alongside any values we derived (clearly flagged as `[derived]`).

## Limitations (non-goals)

We deliberately scope out the following to keep a 48-hour solo build tractable:

- **No reinforcement learning.** Pure stochastic MILP. Easier to reason about, auditable, fast enough.
- **No real-time grid-operator APIs.** All grid events are synthetic; built from challenge-published frequency.
- **No battery degradation cost in v1.** Noted in `docs/assumptions.md §3`.
- **No distribution-transformer / feeder caps.** Inter-vehicle coordination beyond independent per-vehicle solves is out of scope for v1.
- **No mobile app, no auth, no security model.** Dashboard is a local Streamlit app for demo purposes.
- **No model training.** Challenge scopes compute value to *inference only*.

## Repository layout

```
openvpp-orchestrator/
├── README.md                       # this file
├── requirements.txt
├── docs/
│   ├── assumptions.md              # challenge parameters verbatim + derived
│   ├── architecture.md             # component contracts + data flow
│   └── adr/                        # architecture decision records
├── src/
│   ├── fleet/generator.py          # 100-vehicle Phoenix fleet
│   ├── signals/                    # grid_events, lmp, inference_demand
│   ├── forecasting/forecasters.py  # 24h-ahead probabilistic forecasters
│   ├── optimizer/
│   │   ├── milp.py                 # core PuLP formulation
│   │   └── rolling_horizon.py      # MPC replay wrapper
│   ├── simulator/
│   │   ├── backtest.py             # 1-year replay
│   │   └── baselines.py            # 5 strategies
│   └── dashboard/app.py            # Streamlit multi-page demo
├── data/                           # synthetic traces (parquet)
├── results/                        # backtest outputs
├── notebooks/                      # exploratory notebooks
├── deck.pptx                       # 5-slide pitch (generated in Phase 6)
└── reel.mp4                        # 60–90s walkthrough (Phase 6)
```

## License

TBD at submission.
