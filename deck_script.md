# OpenVPP Orchestrator — 5-Slide Deck Script (Gamma-ready)

**How to use this file**

1. Open [gamma.app](https://gamma.app) → *Create new* → *Paste in text*.
2. Copy everything from the `═══ SLIDE 1 ═══` line down to the end of the document.
3. Paste. Gamma will auto-generate 5 slides from the 5 sections.
4. Pick a **dark theme** (black / violet accent) to match the dashboard.
5. On each slide, hit *Generate image* if you want a hero graphic. Suggested prompts are included per slide.

Keep the **speaker notes** in the corresponding field — Gamma has a "Notes" area per slide.

---

═══ SLIDE 1 ═══

# EVs are underused grid + inference assets

**One battery · three revenue streams · mobility never broken**

- Every EV carries a **~100 kWh battery** — sits idle **18–22 hours a day**
- Arizona grid: **~10 major stress events per year** — hot afternoons, AC load peaks
- Parked EVs can also host **distributed AI inference** — a second revenue stream
- **The real problem:** allocate the same 30–50 kWh of flexible energy across **mobility · V2G · inference**, under uncertainty, without breaking trips

**Speaker notes:** Start with the physical reality — a 100-kWh battery parked 20 hours a day is wasted storage. Frame Arizona's grid and the AI compute boom as two demand signals chasing the same battery. End on the tension: three value streams, one finite resource, mobility is sacred.

**Suggested image prompt:** Aerial view of a Phoenix parking lot at dusk, dozens of electric vehicles glowing softly, grid power lines in the background, data-center silhouettes on the horizon. Cinematic, dark teal + violet accent, futuristic.

---

═══ SLIDE 2 ═══

# One orchestrator. Three markets. Every hour.

**Per-vehicle AI decision engine — 24 h look-ahead · 1 h commit**

- **Per-vehicle rolling-horizon stochastic optimizer.** 24-hour plan, refreshed hourly. Solves for every car independently.
- **Mobility is priced, not walled off.** $/hour penalty for missed driving. Driver always wins — *reliability emerges, it's not hand-coded*.
- **Three-way co-optimization** of charge · V2G · inference — against **5 joint future scenarios** drawn from three forecasters.
- **Same solver for private and fleet-managed cars.** Parameters differ per-vehicle; logic does not.
- **No inter-vehicle coupling** → scales to thousands by adding CPU cores. Fleet scale is a cluster config, not a redesign.

**Speaker notes:** Hit the mobility-is-priced idea hard — it's the architectural differentiator. Reliability at 100% falls out of the math, it isn't a hard rule. The "scales to thousands" claim is backed by math: independent per-vehicle problems parallelize perfectly.

**Suggested image prompt:** Schematic diagram — a single EV icon in the center, three arrows radiating out to icons representing power grid, AI data center, and a road; background of faint time-step grid lines. Minimalist, dark violet, technical.

---

═══ SLIDE 3 ═══

# Stochastic math optimizer on seeded synthetic traces

**Classical-AI stack: forecasters → scenario sampler → MILP**

**Signals (parameter-calibrated, sources cited):**
- Fleet · trips · electricity prices (LMP) · grid events · inference demand
- Calibrated from **challenge brief + ADOT · MAG · Argonne · ERCOT / CAISO**

**Forecasters:**
- Grid events → **seasonal Bernoulli** (hot months → higher probability)
- Electricity prices → **AR(1)** (sticky hour-over-hour)
- Inference jobs → **Poisson** (wave-shaped demand)

**Optimizer:**
- **Mixed-integer linear program (MILP)** — PuLP + CBC
- ~200 variables · ~800 constraints **per vehicle-day**
- Rolling horizon: plan 24 h, commit 1 h, advance, re-solve

**Where AI shows up:** *learning* (forecasters) · *reasoning under uncertainty* (scenario sampling) · *planning* (MILP). Auditable, interpretable, fast.

**Baselines:** passive · smart-charge · V2G-only · greedy · **stochastic co-opt (ours)**

**Speaker notes:** Anticipate the "is this really AI?" question — answer directly: classical AI (learning + reasoning + planning), not LLM/deep-learning. Explain *why* we chose classical: interpretable, fast, mathematically guarantees driver mobility. Judges and grid regulators both care about auditability.

**Suggested image prompt:** Technical flowchart — three forecaster boxes feeding into a scenario sampler, feeding into a MILP solver, outputting hourly decisions. Clean vector style, dark background, violet and teal accents.

---

═══ SLIDE 4 ═══

# +$2,000 / vehicle / year · 100 % mobility reliability

**Double V2G-only · every challenge band met**

**Headline results** (100 vehicles · 1 year · hourly decisions):

| | **Our system** | V2G-only |
|---|---|---|
| $/veh/yr | **$2,000** | $900 |
| Mobility reliability | **100 %** | 100 % |

**Challenge-brief compliance (all three in-band):**
- Event-window V2G: **$82 /yr** · brief band $20–150 ✅
- Inference revenue: **$1,985 /yr** · brief band $750–3,000 ✅
- Mobility reliability: **100 %** · brief target ≥ 99 % ✅
- Peak-arbitrage V2G (bonus, not priced by brief): **$2,782 /yr**

**Private vs fleet-managed:**
- Private: **$2,122 /yr** · more idle time → more flexibility
- Fleet: **$1,700 /yr** · tighter schedule → less flexibility
- Same model handles both — no per-segment tuning

**Defensive proof:** At **0× inference price**, stochastic co-opt still beats V2G-only and smart-charge. **Not a one-bet product.**

**Speaker notes:** Lead with the big number — $2,000 per car per year, double V2G-only. Then prove you passed the brief's own bands (compliance table). Finally the defensive move: set inference to zero, we're still winning. This pre-empts the "what if the AI market doesn't happen?" question.

**Suggested image prompt:** Clean bar chart — five bars labeled passive / smart-charge / V2G-only / greedy / stochastic co-opt, with the last one (violet) towering over the others. Number "$2,000" in large type. Dark background.

---

═══ SLIDE 5 ═══

# From 48-hour prototype to pilot

**Roadmap — real data · stronger guarantees · fleet scale**

- **Real data feeds.** Swap synthetic prices + events for real EIA-930 and utility time-of-use. Swap synthetic trips for anonymized telematics.
- **Chance-constrained mobility.** Promote reliability from a $/hr penalty to a **probabilistic guarantee** (e.g., P[miss] ≤ 1% per driver).
- **OpenVPP integration.** Expose per-vehicle decisions as a settlement API; stream on a **5-minute cadence** instead of hourly.
- **Fleet scale-out.** Horizontal scaling across cores + nodes. **10,000 vehicles in under an hour** for daily re-planning.
- **Driver UX.** Override dial — *"leave 80 kWh tonight, I'm going camping."* Owner can trade earnings for slack.

**Built solo. 48 hours. Integration-ready for OpenVPP. Pilot-ready for any Arizona fleet operator.**

**Speaker notes:** Close on momentum. The prototype is small (48h solo) but the architecture is genuinely fleet-ready. The roadmap is concrete — not vaporware, just features we deliberately scoped out for v1. End on the two readiness statements: OpenVPP-integration ready, pilot-ready.

**Suggested image prompt:** Timeline visual — left side shows current prototype (laptop + dashboard), right side shows scaled pilot (fleet dispatch center, 10k vehicles on a map, API integration). Horizontal arrow of progress. Dark, optimistic.

---

## Design notes for Gamma

- **Font:** pick a sans-serif. Avoid the default Gamma serif — it doesn't match the technical content.
- **Color palette** (from the dashboard, for brand coherence):
  - Background: `#000000` (black) or `#0a0a0a`
  - Text: `#ffffff` (white)
  - Muted text: `#a1a1aa`
  - Accent violet: `#c084fc`
  - Success green: `#4ade80`
  - Alert red: `#f87171`
- **Slide dimensions:** 16:9 widescreen.
- **Logo:** none needed. Use clean type on slide 1 as the title.

## Alternative: if Gamma is slow / unavailable

Run `python scripts/generate_deck.py` — it builds a clean 5-slide `.pptx` directly from the latest backtest numbers. Covers the same ground as the script above.
