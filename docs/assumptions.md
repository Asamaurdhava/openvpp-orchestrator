# Assumptions

All numbered parameters below are transcribed **verbatim** from *Software For Energy — OpenVPP Challenge* (the challenge PDF, pages 2–3). Parameters we introduce to make the simulation runnable are flagged `[derived]` with a rationale.

Source: `../Tracks/Software For Energy - OpenVPP Challenge.pdf`

---

## 1. Challenge-specified parameters (verbatim)

### Grid stress events

| Parameter | Unit | Value | Description |
|---|---|---|---|
| Major grid stress events per year | events/year | **10** | Arizona experiences about ten major grid stress events annually. |
| Summer stress events | events/year | **6** | Six of the annual events occur during summer conditions. |
| Winter stress events | events/year | **4** | Four of the annual events occur during winter conditions. |

### Vehicle energy model

| Parameter | Unit | Value | Description |
|---|---|---|---|
| Battery size per vehicle | kWh | **100** | Each EV has a 100 kWh battery. |
| Daily driving consumption | kWh/day | **25–40** | Normal driving consumes between 25 and 40 kWh per day. |
| Required reserve buffer | kWh | **20–30** | Energy preserved for mobility reliability. |
| Flexible usable capacity | kWh | **30–50** | Energy available for grid support or inference after driving + reserve. |
| Idle time | hours/day | **18–22** | Vehicles are idle most of the day. |
| Participation window | hours/day | **8–12** | Part of idle time realistically usable for coordinated participation. |

### V2G economics

| Parameter | Unit | Value | Description |
|---|---|---|---|
| Energy exported per V2G event | kWh/event | **10–30** | Per-vehicle export during a grid event. |
| Peak event electricity price | $/kWh | **0.20–0.50** | V2G export compensation during high-value periods. |
| V2G revenue per event | $/event | **2–15** | Per-vehicle event revenue under the stated assumptions. |
| Annual V2G revenue | $/year | **20–150** | Annual event-driven V2G revenue assuming ~10 events/year. |

### Inference economics

| Parameter | Unit | Value | Description |
|---|---|---|---|
| Inference power demand | kW | **2–8** | Power draw while inference is active. |
| Inference session duration | hours/session | **4–10** | Length of a single session. |
| Inference energy per session | kWh/session | **8–80** | Energy consumed per session. |
| Off-peak charging cost | $/kWh | **0.08–0.15** | Arizona off-peak charging price range. |
| Example inference session energy | kWh/session | **40** | Reference session size for economic reasoning. |
| Example charging cost | $/session | **4.80** | At $0.12/kWh, a 40 kWh session costs $4.80. |
| Inference revenue price | $/kWh equivalent | **0.20–0.60** | Inference priced above electricity cost. |
| Inference revenue per session | $/session | **8–25** | Per-session revenue under the stated assumptions. |
| Inference net profit per session | $/session | **5–20** | Per-session net value after electricity cost. |
| Annual inference revenue | $/year | **750–3,000** | Annual revenue under moderate utilization. |
| Annual inference net profit | $/year | **400–2,000** | Annual net value under moderate utilization. |
| Moderate annual utilization | days/year | **150–250** | Days/year a vehicle participates in inference. |

### Program constraints

| Parameter | Value | Description |
|---|---|---|
| Ownership model | 2 types | Privately owned or fleet managed. |
| Compute scope | inference only | Not model training. |
| Core resource constraint | 30–50 kWh | The same flexible energy pool must be allocated across mobility, V2G, and inference. |

---

## 2. Derived parameters (we introduce these to make the simulation executable)

The challenge PDF intentionally stays at the specification level. To simulate, we must commit to concrete point values within (or consistent with) the published ranges. Each derived value below cites its source or rationale.

### 2.1 Mobility-miss penalty λ `[derived]`

**Value:** `$50 / mobility-failure hour`
**Rationale:** Core differentiator of our approach (ACTION_PLAN.md §3). Encodes driver inconvenience + replacement transport (Uber ≈ $20 + stress premium). Calibrated so that breaking mobility to capture a $10 V2G event is always irrational, but sacrificing trivial participation to protect a high-confidence trip is not. Exposed as a dashboard parameter; defaults to $50.

### 2.2 Charge / discharge power caps `[derived]`

**Charge:** 11 kW (Level 2 home, common US spec).
**Discharge (V2G):** 11 kW (symmetric; bidirectional Level 2 chargers, e.g., Wallbox Quasar, Ford Charge Station Pro).
**Rationale:** PDF does not specify. Level 2 is the residential baseline in Phoenix. Public Level 3 DCFC is out of scope — this is a home/workplace-parked scenario.

### 2.3 Round-trip efficiency `[derived]`

**η_charge:** 0.92 (AC→DC conversion + battery losses)
**η_discharge:** 0.92
**Net V2G round-trip:** ~0.85
**Rationale:** Typical lithium-ion BEV with onboard bidirectional inverter (Argonne ANL reports, 2022–2024).

### 2.4 Hourly LMP (locational marginal price) profile `[derived]`

**Base off-peak:** $0.09/kWh (within challenge range 0.08–0.15).
**Base on-peak (3pm–9pm):** $0.18/kWh.
**Event-hour spike:** $0.35/kWh (within challenge range 0.20–0.50).
**Rationale:** Approximates APS / Salt River Project retail TOU schedules. Challenge uses retail-like pricing for V2G compensation; we carry the same convention.

### 2.5 Fleet composition `[derived]`

**Size:** 100 vehicles.
**Split:** 70 private, 30 fleet-managed.
**Rationale:** Challenge requires "both privately owned vehicles and fleet-managed vehicles." 70/30 reflects current EV ownership mix in Arizona (ADOT 2024 registration data; private dominates). Fleet vehicles get tighter departure schedules and higher λ (fleet SLAs are stricter).

### 2.6 Phoenix commute pattern `[derived]`

**Home dwell:** 18:00–07:30 (weekday).
**Work dwell:** 08:30–17:00 (weekday).
**Weekend dwell:** primarily at home with short trips.
**Trip energy:** sampled from challenge range 25–40 kWh/day, split across 1–3 trips.
**Rationale:** MAG (Maricopa Association of Governments) 2023 commute survey: median weekday commute time 27 min each way; ~60% of workers leave home 7:00–9:00.

### 2.7 Inference demand model `[derived]`

**Generator:** Poisson process with time-varying rate.
**Daytime rate (8am–8pm):** 0.15 session-starts/vehicle/hour.
**Overnight rate (8pm–8am):** 0.05 session-starts/vehicle/hour.
**Rationale:** Inference demand loosely tracks human daytime activity (ChatGPT-style usage peaks 9am–11pm local, per OpenAI usage reports). Tuned so annual inference utilization lands in challenge-stated 150–250 days/vehicle/year.

### 2.8 Grid event timing `[derived]`

**Summer events (6):** late afternoons in June–September, hour 16–19 local.
**Winter events (4):** early mornings in December–February, hour 6–9 local.
**Event duration:** 3–5 hours.
**Rationale:** APS summer peak is driven by AC cooling load around sunset; winter peak is driven by morning heating. Consistent with ERCOT / CAISO stress-event literature scaled to AZ.

### 2.9 Scheduling horizon `[derived]`

**Planning horizon:** 24 hours, 1-hour resolution.
**Re-optimization cadence:** every 1 hour (rolling-horizon MPC).
**Scenarios per stage:** **2**, sampled from the forecasters.
**Rationale:** 24h captures the diurnal cycle and next-day commute commitment. Hourly matches retail energy-market settlement granularity. Original design targeted 5 scenarios per stage (Mohammadi & Mehr, 2023); after benchmarking we dropped to 2 because our rolling-horizon wrapper injects *observed* values at t = 0 into every scenario, so non-anticipativity collapses first-stage variance regardless of scenario count. Going from 5 → 2 cut solve time ~2.5× with no measurable revenue change. The ``forecaster`` module still defaults to ``n_scenarios=5`` — any caller can opt in to the richer sample for a one-off high-fidelity solve.

---

## 3. Explicit non-parameters (things we deliberately do NOT model)

- **Battery degradation cost per cycle.** Priced at 0 in v1. Simplification — real V2G platforms assign ~$0.03–0.08/kWh cycled. Noted in README limitations.
- **Network congestion / distribution-transformer limits.** Treated as infinite. A real OpenVPP deployment must respect LV-feeder caps; out of scope for a 48h build.
- **Demand-response program enrollment fees / fixed payments.** Excluded. Only event-settled revenue is modeled.
- **Real-time grid operator (ISO) APIs.** Not called. All grid events are synthetic.
- **Driver override / manual opt-out.** Assumed always opted-in during participation window.

---

## 4. Update policy

If the user clarifies a range endpoint at build time, update the derived section only — never edit the verbatim tables. Each derived value should keep its rationale one-line.
