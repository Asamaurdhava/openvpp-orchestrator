# OpenVPP Orchestrator — Reel Script (60–90 s)

Record in one take. Screen capture the Streamlit dashboard. Voiceover is first-person.

---

## Beat 1 (0:00–0:10) — Hook

> "Every electric vehicle carries a 100-kilowatt-hour battery that sits unused 18 to 22 hours a day. In Arizona, the grid is under stress. And AI is begging for distributed compute everywhere."

**On screen:** Phoenix map with 100 parked vehicle markers, most idle grey.

---

## Beat 2 (0:10–0:22) — The problem in one sentence

> "We built an AI orchestrator that decides, every hour, for every vehicle, whether it should charge, feed the grid, run inference, or sit idle — without ever breaking the driver's mobility promise."

**On screen:** Switch to the Live Fleet tab. Scrub the snapshot-time slider across 6am → 3pm. Watch colors flip from blue (charge) through grey (idle) to orange (infer) to red (discharge) as LMP rises.

---

## Beat 3 (0:22–0:42) — The money shot

> "Here's what happens during a grid stress event. Watch the fleet flip."

**On screen:** Grid Event Simulator tab. Pick a summer afternoon event (e.g., August 23). The chart shows LMP spiking at the start of the event window. Highlight:
- Blue charge bars **before** the event
- Red discharge bars **during** the event
- The MWh delivered metric updating live

> "The fleet pre-charges, then dumps 2 megawatt-hours into the grid during the stress hours. The optimizer saw the event coming in the forecast scenarios and planned around it."

---

## Beat 4 (0:42–1:00) — The defensive proof

> "Judges always ask: what if the EV-inference market doesn't materialize? We priced that in."

**On screen:** Economics Sensitivity tab. Drag the slider from 1.0 down to 0.0.

> "At zero inference revenue, the system falls back to pure V2G plus smart charging — still profitable, still working, still valuable. That's not a single-bet product."

---

## Beat 5 (1:00–1:15) — The headline

**On screen:** Baseline Comparison tab. Point at the five bars.

> "Stochastic co-optimization: thousands of dollars per vehicle per year. Beats the V2G-only baseline by about 2x. Beats dumb passive charging — which actually loses money — by everything. And it holds 100 percent mobility reliability across the year."

---

## Beat 6 (1:15–1:25) — The close

> "Integration-ready for OpenVPP. Pilot-ready for any fleet operator in Arizona. Built in 48 hours — solo."

**On screen:** Repo tree briefly, then GitHub URL.

---

## Record tips

- One take. Don't stop-start.
- QuickTime (Cmd+Shift+5, full-screen record with microphone). Trim only the head and tail.
- Pre-open the Streamlit app with data loaded. Don't wait for Streamlit to rerun mid-reel.
- Pre-pick the event you're going to demo so the selectbox is already correct.
- Keep voice calm — read the script, don't riff. Pace = ~150 wpm.
