# OpenVPP Orchestrator

**One EV battery · three revenue streams · every hour, optimally allocated — under uncertainty, without breaking the driver's trip.**

Submission · **OpenVPP Challenge · Software For Energy · Arizona track**

---

## At a glance

| Metric | Our system |
|---|---|
| Extra revenue per vehicle, per year | **≈ $2,000** |
| Lift vs. V2G-only baseline | **+100%** |
| Mobility reliability (driver never stranded) | **100%** |
| Still profitable if AI-inference price → $0 | **Yes** |
| Works identically for 1 car or 10,000 | **Yes** (per-car math, no inter-vehicle coupling) |
| Built in | **48 hours · solo** |

---

## Three-line summary

1. **Problem:** Every electric car carries a ~100 kWh battery that sits idle 18–22 hours a day. It *could* earn money three ways — selling power back to the grid, running AI jobs, and charging cheaply — but can't do all three at once without risking the driver's next trip.
2. **Solution:** An AI orchestrator that decides, **for every car, every hour**, the best mix of *charge · discharge to grid · run inference · sit idle* — looking 24 hours ahead, under uncertainty.
3. **Result:** Beats every simpler baseline. Every challenge target met. Still profitable even if the AI-inference market never materializes.

---

## Quickstart (3 minutes)

```bash
git clone https://github.com/Asamaurdhava/openvpp-orchestrator.git
cd openvpp-orchestrator
make install        # create virtual environment + install packages
make dash           # launch the live demo dashboard
```

Dashboard opens at [http://localhost:8501](http://localhost:8501). **Four tabs:**

| Tab | What to look for |
|---|---|
| **Live Fleet** | 100-car Phoenix map. Pick any date + hour → every car's decision updates live. |
| **Grid Event Simulator** | Pick a real-dated stress event (e.g., Aug 23) → fleet pre-charges, then discharges into the grid. |
| **Economics Sensitivity** | Click `1.0×` → `0.5×` → `0×` on inference price → watch us stay profitable. |
| **Baseline Comparison** | Our system vs. 4 simpler strategies. Challenge-brief compliance row. Private vs. fleet-managed split. |

---

## The problem — in plain English

Electric cars have huge batteries (~100 kWh each — enough to power a home for 2–3 days). But the cars are parked most of the time. That battery could be earning money three different ways:

- **Selling energy back to the grid** during hot afternoons when Arizona's grid is stressed (called "V2G" or *vehicle-to-grid*).
- **Running AI jobs.** Modern EVs have real computer chips onboard. Parked EVs could run distributed AI work and get paid for it.
- **Just charging cheaply** overnight when electricity is abundant, to cut the owner's bill.

The catch: **all three compete for the same limited battery energy**, and the driver still needs the car to just… drive them places. Nobody had a platform that juggles all three without risking the driver's trip.

We built that orchestrator.

---

## How it works — the three design choices

### 1. We *price* the driver's trip instead of locking it in with a hard rule

The naive approach: force the battery to always be ≥ 80% charged at 7 AM. That works but leaves money on the table — sometimes the driver doesn't leave until 10 AM, sometimes the market is paying a fortune for energy right now.

Our approach: attach a **dollar penalty** to every missed hour of driving (≈ $50/hr, tunable per driver). The optimizer weighs that cost against grid + AI revenue. Because a missed trip is really expensive, the optimizer almost always decides to protect mobility — and reliability ends up at 100% without us hand-coding it.

### 2. We plan against **multiple possible futures**, not one guess

We don't know exactly when the grid will stress, what prices will be, or when AI jobs will show up. So the optimizer runs against **5 different plausible futures** at once, and picks the action that does well across all of them. This is called *stochastic programming* in the textbooks.

### 3. Three-way co-optimization with graceful failure

The brief mentions two value streams (V2G and compute) that compete for the same battery. We stack them. If the AI-inference market disappears tomorrow, the system automatically falls back to pure V2G + smart-charging — still profitable, still valuable. **Not a one-bet product.** You can verify this in the dashboard: set inference price to `0×` and we still beat every simpler baseline.

---

## Technical approach (for readers who want the math)

- **Per-car mathematical optimizer** — a **mixed-integer linear program (MILP)** solved with PuLP + CBC. Each car has ≈ 200 decision variables and ≈ 800 constraints per day.
- **Three probabilistic forecasters** feed joint future scenarios into every decision:
  - Grid stress events → seasonal Bernoulli (hot months → higher event probability)
  - Electricity prices (LMP) → AR(1) time series (sticky prices)
  - AI-inference job arrivals → Poisson (wave-shaped demand)
- **Rolling-horizon replay** — at every hour, re-solve a fresh 24-hour-ahead plan, commit to only the first hour's action, advance the clock, re-solve. This is how model-predictive control works.
- **Parallel across the fleet** — the per-car problem has no inter-vehicle coupling, so fleet scale = `(cost per car) × (N cars) / (CPU cores)`. Scaling to "thousands of vehicles" is a cluster-config choice, not an architectural change.

See [docs/architecture.md](docs/architecture.md) for the full component diagram and decision-variable list.

---

## Where AI shows up

The brief asks for an *"AI-driven orchestration system."* Three distinct AI techniques are used — the classical-AI kind, not LLM/deep-learning:

1. **Learning** — three forecasting models learn patterns from data (hot afternoons → grid events; sticky prices; wave-shaped inference demand).
2. **Reasoning under uncertainty** — instead of one guess, the system samples 5 possible futures and reasons across them.
3. **Planning / optimization** — classical mathematical optimization picks the best action respecting every constraint.

We deliberately avoided neural networks: too little data, not interpretable, slower than classical solvers for this shape of problem. Every decision the system makes can be traced to a constraint or a dual price — that matters to grid regulators and to drivers.

---

## Results

### Fleet-wide (1 year, 100 vehicles, hourly decisions)

| Strategy | $/veh/yr | Mobility reliability | Notes |
|---|---|---|---|
| `passive` (charge whenever plugged) | ≈ $0 | 100% | Baseline — no market participation |
| `smart_charge` | modest | 100% | Cheap-hour charging only |
| `v2g_only` | ≈ $900 | 100% | Classic V2G platform, no inference |
| `greedy` (myopic) | worse | lower | No forecasting — chases the spot market |
| **`stochastic_coopt`** (ours) | **≈ $2,000** | **100%** | Full three-way co-optimization |

### Challenge-brief compliance

| Brief target | Brief band | Our number | In band? |
|---|---|---|---|
| Event-window V2G | $20–150 /veh/yr | ≈ $82 | ✅ |
| Inference revenue | $750–3,000 /veh/yr | ≈ $1,985 | ✅ |
| Mobility reliability | ≥ 99% | 100% | ✅ |
| Peak-arbitrage V2G *(bonus)* | not priced by brief | ≈ $2,782 | — (flagged as extra value) |

### Private vs. fleet-managed vehicles

| Ownership | $/veh/yr | Why |
|---|---|---|
| Private | ≈ $2,122 | More idle time → more flexibility to monetize |
| Fleet-managed | ≈ $1,700 | Tighter trip schedule → less flexibility |

Same model handles both — just different parameters per vehicle.

---

## What's real vs. synthetic

| Data | Source | Status |
|---|---|---|
| Phoenix hourly temperature (2025) | NOAA via Open-Meteo Archive | **Real** — used in the Event Simulator to prove synthetic summer events land on genuinely hot days |
| Fleet (100 vehicles, Phoenix metro) | Generated from [ADOT](https://azdot.gov) + [MAG](https://azmag.gov) commute patterns | Synthetic, parameter-calibrated |
| Driver trip patterns | Generated from Argonne National Lab travel surveys | Synthetic, parameter-calibrated |
| Electricity prices (LMP) | Generated to match ERCOT / CAISO distributions | Synthetic, parameter-calibrated |
| Grid stress events | ≈ 10/year · seasonal · brief-specified | Synthetic, brief-calibrated |
| AI-inference job demand | Poisson process · brief-specified price band | Synthetic, brief-calibrated |
| MILP solver | PuLP + CBC (open source) | Real code, real math |
| Dashboard | Streamlit | Real code |

**Why synthetic is the right call:** the challenge tests your *design and decision logic*, not your access to Arizona's live grid feed. Every parameter is either copied from the brief or cited from a public source (ADOT, MAG, Argonne, ERCOT, CAISO). The full parameter sheet is [docs/assumptions.md](docs/assumptions.md), with every derived value flagged.

---

## Full pipeline — reproduce from scratch

```bash
make install          # venv + pip install       (~2 min)
make reproduce        # data + backtest + deck   (~30 min)
make dash             # launch Streamlit
```

Or step-by-step:

```bash
# 1. Generate synthetic fleet + 1-year signal traces
python -m src.fleet.generator
python -m src.signals.grid_events
python -m src.signals.lmp
python -m src.signals.inference_demand
python -m src.sanity_check

# 2. Quick visual sanity of the forecasters (optional)
python -m src.forecasting.demo

# 3. Single-vehicle optimizer demo (optional, ~10 s)
python -m src.optimizer.demo

# 4. Full backtest across 5 strategies
python -m src.simulator.backtest --n-vehicles 10 --horizon-days 365

# 5. Download real Phoenix temperature (optional, ~10 s)
python scripts/fetch_real_references.py

# 6. Launch the dashboard
streamlit run src/dashboard/app.py
```

---

## Assumptions

Every parameter used in the simulation lives in [docs/assumptions.md](docs/assumptions.md). Each entry is tagged:

- **[from brief]** — copied verbatim from the OpenVPP Challenge PDF.
- **[derived]** — reasoned from a named public source (ADOT, MAG, Argonne, ERCOT, CAISO, etc.), with the rationale recorded alongside it.

This is deliberate. When a judge asks *"where did this number come from?"* there's one answer: open the assumptions doc.

---

## Limitations (honest non-goals for v1)

- **No reinforcement learning.** Classical optimization is auditable, provably safe for mobility, and fast enough.
- **No live grid-operator feed.** All grid events are synthetic, calibrated to brief frequency (~10/year).
- **No battery-degradation cost in v1.** Flagged in assumptions; trivial to add as a linear term.
- **No distribution-transformer or feeder caps.** The v1 MILP is independent per-vehicle.
- **No mobile app, no auth, no production security model.** Dashboard is a local Streamlit demo.
- **No model training on-vehicle.** The challenge scopes EV compute to inference only.

---

## Repository layout

```
openvpp-orchestrator/
├── README.md                       # this file
├── requirements.txt
├── Makefile                        # install · reproduce · dash
├── docs/
│   ├── assumptions.md              # every parameter + its source
│   ├── architecture.md             # component diagram + data flow
│   └── adr/                        # architecture decision records
├── src/
│   ├── fleet/generator.py          # 100-vehicle Phoenix fleet
│   ├── signals/                    # grid_events · lmp · inference_demand
│   ├── forecasting/forecasters.py  # 24-hour-ahead probabilistic forecasters
│   ├── optimizer/
│   │   ├── milp.py                 # core PuLP formulation
│   │   └── rolling_horizon.py      # model-predictive control wrapper
│   ├── simulator/
│   │   ├── backtest.py             # 1-year replay engine
│   │   └── baselines.py            # 5 strategies
│   └── dashboard/app.py            # Streamlit 4-tab demo
├── scripts/
│   └── fetch_real_references.py    # Phoenix temp from NOAA (Open-Meteo)
├── data/                           # synthetic traces (committed for clone-and-run)
└── results/                        # backtest outputs (committed)
```

---

## Submission artifacts

This repository is the **codebase** for the submission. The non-code artifacts (pitch deck, reel video) are bundled separately with the submission:

| Required by brief | Where |
|---|---|
| GitHub repo (codebase) | [github.com/Asamaurdhava/openvpp-orchestrator](https://github.com/Asamaurdhava/openvpp-orchestrator) |
| README | This file |
| 5-slide pitch deck | Submitted alongside the repo link |
| 60–90s reel | Submitted alongside the repo link |
| Sample data · dashboards | `data/*.parquet` · `streamlit run src/dashboard/app.py` |

---

## Contact

Built for the OpenVPP Challenge · 2026.
Questions: **vrajput5@asu.edu**
