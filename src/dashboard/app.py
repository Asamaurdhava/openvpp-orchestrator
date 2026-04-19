"""OpenVPP Orchestrator — Streamlit demo dashboard.

Design principles
-----------------
* **One primary element per view.** Everything else supports it.
* **Typography as hierarchy**, not borders and pills. Cards only where the
  content truly needs a container (the single hero block).
* **Inline text rows** for supporting KPIs — "42 charging · 18 V2G · 7 idle"
  is easier to scan than four bordered cards.
* **Whitespace does the separating.** Section titles introduce, not
  decorate.

Run: ``streamlit run src/dashboard/app.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

STRATEGY_ORDER = ["passive", "smart_charge", "v2g_only", "greedy", "stochastic_coopt"]
STRATEGY_LABELS = {
    "passive": "Passive",
    "smart_charge": "Smart charge",
    "v2g_only": "V2G only",
    "greedy": "Greedy",
    "stochastic_coopt": "Stochastic co-opt",
}
# Catppuccin-Mocha inspired palette — muted, warm, easy on eyes in dark mode
STRATEGY_COLORS = {
    "passive":          "#7f849c",  # overlay 0 (dim slate)
    "smart_charge":     "#89b4fa",  # blue (soft sky)
    "v2g_only":         "#94e2d5",  # teal
    "greedy":           "#f9e2af",  # yellow (soft butter)
    "stochastic_coopt": "#f38ba8",  # pink-red (muted)
}
ACTION_COLORS = {
    "idle":      [108, 112, 134, 180],   # surface2
    "charge":    [137, 180, 250, 230],   # blue
    "discharge": [243, 139, 168, 230],   # pink
    "infer":     [250, 179, 135, 230],   # peach
}

# Core palette tokens
BG_BASE    = "#1e1e2e"   # main app background
BG_MANTLE  = "#181825"   # sidebar / deepest
BG_SURFACE = "#313244"   # cards, hero, plot bg
BG_OVERLAY = "#45475a"   # borders, separators

TEXT       = "#cdd6f4"   # primary (cream off-white — not harsh)
TEXT_MUTED = "#a6adc8"   # secondary
TEXT_DIM   = "#7f849c"   # tertiary

ACCENT     = "#f38ba8"   # pink-red (stochastic, alerts)
ACCENT_DIM = "#eba0ac"
WARN       = "#f9e2af"   # yellow (LMP line, warnings)
OK         = "#a6e3a1"   # green (on-target indicators)
SKY        = "#89b4fa"   # blue (charge)


# ---------- Styling ----------

def inject_css() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://api.fontshare.com" crossorigin>
        <link href="https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@300,400,500,600,700,800,900&display=swap" rel="stylesheet">

        <style>
        /* ---- Font: Cabinet Grotesk everywhere ---- */
        html, body, .stApp, [class*="css"], [class*="st-"],
        button, input, select, textarea,
        h1, h2, h3, h4, h5, h6, p, span, div, label, li, a, code {{
            font-family: 'Cabinet Grotesk', 'Inter', system-ui, -apple-system,
                         BlinkMacSystemFont, sans-serif !important;
        }}
        code {{
            font-family: 'Cabinet Grotesk', ui-monospace, SFMono-Regular,
                         Menlo, Consolas, monospace !important;
            font-weight: 500;
        }}

        /* Base app background — soft dark blue-purple, not black */
        .stApp {{background: {BG_BASE};}}

        /* Layout */
        .block-container {{padding: 2rem 2.5rem 4rem; max-width: 1280px;}}

        /* Hide Streamlit chrome */
        #MainMenu, footer, header[data-testid="stHeader"] {{visibility: hidden; height: 0;}}

        /* Headings */
        .brand-title {{
            font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em;
            color: {TEXT}; margin: 0;
        }}
        .brand-sub {{color: {TEXT_MUTED}; font-size: 0.92rem;
                    margin: 0.15rem 0 2rem 0;}}

        .section-label {{
            color: {TEXT_MUTED}; font-size: 0.78rem; letter-spacing: 0.04em;
            margin: 2.2rem 0 0.55rem 0; font-weight: 500;
        }}
        .tab-lead {{
            color: {TEXT}; font-size: 0.95rem; line-height: 1.55;
            margin: 0.3rem 0 1.6rem 0; max-width: 780px;
        }}

        /* Hero block */
        .hero {{
            background: linear-gradient(135deg, {BG_SURFACE} 0%, {BG_MANTLE} 100%);
            border-radius: 14px; padding: 1.8rem 2rem; margin-bottom: 2.5rem;
            border: 1px solid {BG_OVERLAY};
        }}
        .hero-value {{
            font-size: 3.2rem; font-weight: 700; letter-spacing: -0.03em;
            color: {TEXT}; line-height: 1; margin: 0.35rem 0 0.4rem 0;
        }}
        .hero-value-accent {{color: {ACCENT};}}
        .hero-caption {{color: {TEXT_MUTED}; font-size: 0.82rem;
                       letter-spacing: 0.06em; margin-bottom: 0.2rem;}}
        .hero-sub {{color: {TEXT}; font-size: 0.98rem;}}
        .hero-sub strong {{color: {TEXT}; font-weight: 700;}}

        /* Inline stat rows */
        .stat-row {{
            color: {TEXT}; font-size: 1rem; margin: 0.3rem 0 0.8rem 0;
        }}
        .stat-row strong {{color: {TEXT}; font-weight: 700;}}
        .stat-row .sep {{color: {BG_OVERLAY}; margin: 0 0.6rem;}}
        .stat-row .accent {{color: {ACCENT}; font-weight: 700;}}

        /* Pills — used sparingly */
        .pill {{
            display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
            background: {BG_OVERLAY}; color: {TEXT_MUTED}; font-size: 0.7rem;
            letter-spacing: 0.04em; font-weight: 600; vertical-align: middle;
        }}
        .pill-alert {{background: #583845; color: {ACCENT};}}
        .pill-ok {{background: #3d5050; color: {OK};}}

        /* Callouts */
        .callout {{
            background: transparent;
            border-left: 2px solid {BG_OVERLAY};
            padding: 0.2rem 1rem; color: {TEXT_MUTED};
            font-size: 0.88rem; line-height: 1.55;
            margin: 1.4rem 0 0 0;
        }}
        .callout strong {{color: {TEXT};}}

        /* Streamlit metrics fallback */
        div[data-testid="stMetricLabel"] {{color: {TEXT_MUTED} !important; font-size: 0.75rem;}}
        div[data-testid="stMetricValue"] {{color: {TEXT} !important;}}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0; border-bottom: 1px solid {BG_OVERLAY}; padding-bottom: 0;
            margin-bottom: 1.6rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent; color: {TEXT_MUTED};
            padding: 0.7rem 1.3rem; border-radius: 0;
            font-weight: 500; border-bottom: 2px solid transparent;
            margin-right: 0.25rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: transparent !important; color: {TEXT} !important;
            border-bottom: 2px solid {ACCENT} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {BG_MANTLE}; border-right: 1px solid {BG_OVERLAY};
        }}
        section[data-testid="stSidebar"] * {{color: {TEXT};}}
        section[data-testid="stSidebar"] code {{
            background: {BG_SURFACE}; color: {WARN}; padding: 0.05rem 0.3rem;
            border-radius: 4px; font-size: 0.82rem;
        }}

        /* Slider recolor */
        .stSlider [data-baseweb="slider"] > div > div {{background: {ACCENT} !important;}}

        /* Selectbox */
        .stSelectbox > div > div {{
            background: {BG_SURFACE}; border-color: {BG_OVERLAY};
        }}

        /* Expander */
        div[data-testid="stExpander"] {{
            background: {BG_MANTLE}; border: 1px solid {BG_OVERLAY};
            border-radius: 8px;
        }}

        /* Remove extra spacing around plots */
        div[data-testid="stPlotlyChart"] {{margin-bottom: 0;}}

        /* Dataframe */
        div[data-testid="stDataFrame"] {{background: {BG_SURFACE}; border-radius: 8px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_SURFACE,
    font=dict(family="'Cabinet Grotesk', Inter, system-ui, sans-serif", size=13, color=TEXT),
    margin=dict(l=55, r=25, t=35, b=50),
    xaxis=dict(gridcolor=BG_OVERLAY, zerolinecolor=BG_OVERLAY, linecolor=BG_OVERLAY),
    yaxis=dict(gridcolor=BG_OVERLAY, zerolinecolor=BG_OVERLAY, linecolor=BG_OVERLAY),
    legend=dict(bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color=TEXT, family="'Cabinet Grotesk', Inter, sans-serif")),
)


def themed(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ---------- Data ----------

@st.cache_data
def load_fleet() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "fleet.parquet")


@st.cache_data
def load_grid_events() -> pd.DataFrame:
    g = pd.read_parquet(DATA_DIR / "grid_events.parquet")
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    return g


@st.cache_data
def load_lmp() -> pd.DataFrame:
    l = pd.read_parquet(DATA_DIR / "lmp.parquet")
    l["timestamp"] = pd.to_datetime(l["timestamp"])
    return l


@st.cache_data
def load_backtest_summary() -> pd.DataFrame | None:
    p = RESULTS_DIR / "backtest_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


@st.cache_data
def load_backtest_metrics() -> pd.DataFrame | None:
    p = RESULTS_DIR / "backtest_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_decisions(strategy: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / f"backtest_decisions_{strategy}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


# ---------- UI primitives ----------

def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def tab_lead(text: str) -> None:
    st.markdown(f'<div class="tab-lead">{text}</div>', unsafe_allow_html=True)


def stat_row(parts: list[str]) -> None:
    html = '<span class="sep">·</span>'.join(parts)
    st.markdown(f'<div class="stat-row">{html}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def missing_backtest() -> None:
    callout(
        "<strong>Backtest not yet run.</strong> Execute "
        "<code>python -m src.simulator.backtest</code> to populate this view."
    )


def localize_like(reference: pd.Series, ts: pd.Timestamp) -> pd.Timestamp:
    tz = reference.dt.tz
    ts = pd.Timestamp(ts)
    if ts.tz is None and tz is not None:
        return ts.tz_localize(tz)
    if ts.tz is not None and tz is not None:
        return ts.tz_convert(tz)
    return ts


# ---------- Hero ----------

def render_hero(summary: pd.DataFrame | None, fleet: pd.DataFrame, grid: pd.DataFrame) -> None:
    """Single big-number hero block — one primary metric, a short supporting line."""
    n_events = int(grid["event_id"].dropna().nunique())

    if summary is not None and "stochastic_coopt" in summary.index:
        stoch = summary.loc["stochastic_coopt"]
        rev = stoch["total_revenue_usd"]
        rel = stoch["mobility_reliability"] * 100
        support = []
        if "v2g_only" in summary.index:
            delta = rev - summary.loc["v2g_only", "total_revenue_usd"]
            pct = delta / max(abs(summary.loc["v2g_only", "total_revenue_usd"]), 1) * 100
            support.append(f"<strong>{pct:+.0f}%</strong> vs V2G-only")
        support.append(f"<strong>{rel:.1f}%</strong> mobility reliability")
        support.append(f"{len(fleet)} vehicles · 1-year backtest")
        st.markdown(
            f"""
            <div class="hero">
              <div class="hero-caption">STOCHASTIC CO-OPT · ANNUAL REVENUE PER VEHICLE</div>
              <div class="hero-value hero-value-accent">${rev:,.0f}</div>
              <div class="hero-sub">{' · '.join(support)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hero">
              <div class="hero-caption">SETUP</div>
              <div class="hero-value">{len(fleet)} vehicles</div>
              <div class="hero-sub">
                {n_events} grid events · 1-year simulation ·
                <span class="pill">backtest pending</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Tabs ----------

def tab_live_fleet(fleet: pd.DataFrame, lmp: pd.DataFrame, grid: pd.DataFrame) -> None:
    tab_lead(
        "Each marker is one vehicle. Color shows what the orchestrator has the vehicle "
        "doing right now. Scrub through the year to see how dispatch shifts."
    )

    decisions = load_decisions("stochastic_coopt")
    if decisions is None:
        missing_backtest()
        return

    min_ts = decisions["timestamp"].min()
    max_ts = decisions["timestamp"].max()
    events = grid[grid["is_event"]]
    default = (
        events["timestamp"].iloc[len(events) // 2]
        if not events.empty else (min_ts + (max_ts - min_ts) / 3)
    )

    c_slider, c_state = st.columns([3, 2])
    with c_slider:
        pick = st.slider(
            "Snapshot time",
            min_value=min_ts.to_pydatetime(),
            max_value=max_ts.to_pydatetime(),
            value=default.to_pydatetime(),
            step=pd.Timedelta(hours=1).to_pytimedelta(),
            format="MMM D · HH:mm",
            label_visibility="collapsed",
        )
    pick_ts = localize_like(decisions["timestamp"], pick)

    current_lmp_row = lmp[lmp["timestamp"] == pick_ts]
    current_lmp = float(current_lmp_row["lmp_usd_per_kwh"].iloc[0]) if not current_lmp_row.empty else np.nan
    ev_row = grid[grid["timestamp"] == pick_ts]
    is_event_now = bool(ev_row["is_event"].iloc[0]) if not ev_row.empty else False
    pill_html = (
        '<span class="pill pill-alert">GRID EVENT</span>' if is_event_now
        else '<span class="pill">normal</span>'
    )

    with c_state:
        st.markdown(
            f"""
            <div style="text-align:right; padding-top:0.15rem;">
              <div style="color:{TEXT_MUTED}; font-size:0.78rem;">{pick_ts:%A · %b %d · %H:%M}</div>
              <div style="color:{TEXT}; font-size:1.4rem; font-weight:700; letter-spacing:-0.01em;">
                ${current_lmp:.3f}/kWh
              </div>
              <div>{pill_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    snapshot = decisions[decisions["timestamp"] == pick_ts].merge(
        fleet[["vehicle_id", "home_lat", "home_lon", "ownership"]],
        on="vehicle_id", how="inner",
    )

    def action_for(row) -> str:
        if row["charge_kw"] > 0.1:
            return "charge"
        if row["discharge_kw"] > 0.1:
            return "discharge"
        if row["infer_active"]:
            return "infer"
        return "idle"

    snapshot["action"] = snapshot.apply(action_for, axis=1)
    snapshot["color"] = snapshot["action"].map(ACTION_COLORS)

    # Inline count row
    counts = snapshot["action"].value_counts().to_dict()
    stat_row([
        f'<strong>{counts.get("charge", 0)}</strong> charging',
        f'<strong>{counts.get("discharge", 0)}</strong> discharging (V2G)',
        f'<strong>{counts.get("infer", 0)}</strong> running inference',
        f'<strong>{counts.get("idle", 0)}</strong> idle',
    ])

    # Map
    view = pdk.ViewState(
        latitude=float(snapshot["home_lat"].mean()) if len(snapshot) else 33.5,
        longitude=float(snapshot["home_lon"].mean()) if len(snapshot) else -112.0,
        zoom=9.2, pitch=30, bearing=-20,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=snapshot,
        get_position=["home_lon", "home_lat"],
        get_fill_color="color",
        get_radius=500,
        pickable=True, opacity=0.9, stroked=True,
        get_line_color=[24, 24, 37], line_width_min_pixels=1,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer], initial_view_state=view, map_style="dark",
        tooltip={"text": "{vehicle_id} · {action}\nSoC {soc_kwh} kWh"},
    ))

    # Legend — minimal
    legend = []
    for k, c in [("charge", ACTION_COLORS["charge"]), ("discharge", ACTION_COLORS["discharge"]),
                 ("infer", ACTION_COLORS["infer"]), ("idle", ACTION_COLORS["idle"])]:
        col = f"rgba({c[0]},{c[1]},{c[2]},0.9)"
        legend.append(
            f'<span style="display:inline-flex; align-items:center; margin-right:1.4rem;">'
            f'<span style="display:inline-block; width:9px; height:9px; border-radius:50%; '
            f'background:{col}; margin-right:0.45rem;"></span>{k}</span>'
        )
    st.markdown(
        f'<div style="color:{TEXT_DIM}; font-size:0.82rem; margin-top:0.6rem;">{"".join(legend)}</div>',
        unsafe_allow_html=True,
    )

    # SoC distribution — small, secondary
    section_label("STATE OF CHARGE DISTRIBUTION")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=snapshot["soc_kwh"], nbinsx=20,
        marker=dict(color=SKY, line=dict(color=BG_MANTLE, width=1)),
    ))
    fig.add_vline(x=27.5, line=dict(color=ACCENT, dash="dash"),
                  annotation_text="reserve", annotation_position="top")
    fig.update_layout(height=200, xaxis_title="SoC (kWh)", yaxis_title="vehicles")
    st.plotly_chart(themed(fig), use_container_width=True)


def tab_event_simulator(grid: pd.DataFrame, lmp: pd.DataFrame) -> None:
    tab_lead(
        "Pick a grid stress event. See how the fleet pre-charged before the window and "
        "dumped power into the grid during — all without a hand-coded rule."
    )

    decisions = load_decisions("stochastic_coopt")
    if decisions is None:
        missing_backtest()
        return

    event_options = (
        grid[grid["is_event"]]
        .groupby("event_id")
        .agg(start=("timestamp", "min"), end=("timestamp", "max"),
             severity=("severity", "first"))
        .reset_index()
    )
    event_options["label"] = event_options.apply(
        lambda r: f"Event {int(r['event_id']):02d} · {r['start']:%b %d, %Y} · "
                  f"{(r['end']-r['start']).total_seconds()/3600+1:.0f}h · severity {r['severity']:.2f}",
        axis=1,
    )

    # Group by season: winter = Dec/Jan/Feb, summer = Jun-Sep
    winter_mask = event_options["start"].dt.month.isin([12, 1, 2])
    winter = event_options[winter_mask].sort_values("start")["label"].tolist()
    summer = event_options[~winter_mask].sort_values("start")["label"].tolist()

    # Non-selectable season dividers shown inside the dropdown for context only
    DIV_WINTER = "─────  WINTER  ─────"
    DIV_SUMMER = "─────  SUMMER  ─────"
    options = [DIV_WINTER] + winter + [DIV_SUMMER] + summer
    default_idx = 1 if winter else (2 if summer else 0)

    raw = st.selectbox("Event", options=options, index=default_idx,
                       label_visibility="collapsed", key="event_selectbox")

    # If the user clicks a divider, gracefully fall back to the nearest real event
    if raw in (DIV_WINTER, DIV_SUMMER):
        chosen = winter[0] if (raw == DIV_WINTER and winter) else summer[0]
    else:
        chosen = raw

    chosen_row = event_options[event_options["label"] == chosen].iloc[0]

    # Compact summary line
    duration_h = int((chosen_row["end"] - chosen_row["start"]).total_seconds() / 3600) + 1
    st.markdown(
        f'<div class="stat-row">'
        f'<strong>{chosen_row["start"]:%A, %b %d}</strong> '
        f'<span class="sep">·</span> {chosen_row["start"]:%H:%M}–{chosen_row["end"]:%H:%M} '
        f'<span class="sep">·</span> <strong>{duration_h}h</strong> duration '
        f'<span class="sep">·</span> <strong>{chosen_row["severity"]:.2f}</strong> severity'
        f'</div>',
        unsafe_allow_html=True,
    )

    window_start = chosen_row["start"] - pd.Timedelta(hours=8)
    window_end = chosen_row["end"] + pd.Timedelta(hours=8)
    lmp_win = lmp[(lmp["timestamp"] >= window_start) & (lmp["timestamp"] <= window_end)]
    dec_win = decisions[(decisions["timestamp"] >= window_start) & (decisions["timestamp"] <= window_end)]

    fleet_action = (
        dec_win.groupby("timestamp")
        .agg(charge_total=("charge_kw", "sum"), discharge_total=("discharge_kw", "sum"))
        .reset_index()
    )

    # Primary: big chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fleet_action["timestamp"], y=-fleet_action["charge_total"],
        name="Charge", yaxis="y2", marker_color=SKY, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=fleet_action["timestamp"], y=fleet_action["discharge_total"],
        name="Discharge", yaxis="y2", marker_color=ACCENT, opacity=0.9,
    ))
    fig.add_trace(go.Scatter(
        x=lmp_win["timestamp"], y=lmp_win["lmp_usd_per_kwh"],
        name="LMP ($/kWh)", yaxis="y1", line=dict(color=WARN, width=2.5),
    ))
    fig.add_vrect(
        x0=chosen_row["start"], x1=chosen_row["end"],
        fillcolor="rgba(243,139,168,0.15)", line_width=0,
        annotation_text="EVENT", annotation_position="top left",
        annotation=dict(font_color=ACCENT_DIM, font_size=11),
    )
    fig.update_layout(
        yaxis=dict(title="LMP ($/kWh)", side="right"),
        yaxis2=dict(title="Fleet power (kW)", overlaying="y", anchor="x",
                    showgrid=False, side="left"),
        barmode="relative", height=480,
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    st.plotly_chart(themed(fig), use_container_width=True)

    # Supporting stats inline
    in_event = dec_win[
        (dec_win["timestamp"] >= chosen_row["start"]) & (dec_win["timestamp"] <= chosen_row["end"])
    ]
    total_mwh = in_event["discharge_kw"].sum() / 1000.0
    inlmp = lmp.set_index("timestamp")["lmp_usd_per_kwh"]
    in_event = in_event.assign(lmp=in_event["timestamp"].map(inlmp))
    event_revenue = float((in_event["discharge_kw"] * in_event["lmp"]).sum())
    dispatched = int(in_event[in_event["discharge_kw"] > 0.1]["vehicle_id"].nunique())
    total_vehicles = decisions["vehicle_id"].nunique()

    stat_row([
        f'<span class="accent">{total_mwh:.2f} MWh</span> delivered',
        f'<strong>${event_revenue:,.2f}</strong> V2G revenue',
        f'<strong>{dispatched}</strong> of <strong>{total_vehicles}</strong> vehicles dispatched',
    ])

    callout(
        "<strong>The demo moment.</strong> The optimizer sampled the event in its 24h forecast "
        "scenarios, pre-charged during the hours before, and dispatched into the grid the "
        "moment the window opened — without any hand-coded rule."
    )


def tab_sensitivity() -> None:
    tab_lead(
        "What if the EV-inference market never materializes? Drag the slider to zero and "
        "watch the platform fall back to a working V2G + smart-charging system."
    )

    metrics = load_backtest_metrics()
    if metrics is None:
        missing_backtest()
        return

    multiplier = st.slider(
        "Inference revenue multiplier",
        min_value=0.0, max_value=2.0, value=1.0, step=0.05,
        help="1.0 = prices as published. 0.0 = no inference market.",
    )

    m = metrics.copy()
    m["counterfactual"] = (
        m["total_revenue_usd"]
        - m["inference_revenue_usd"]
        + multiplier * m["inference_revenue_usd"]
    )
    summary = (
        m.groupby("strategy")["counterfactual"].mean()
        .reindex(STRATEGY_ORDER).dropna()
    )

    stoch_val = float(summary.get("stochastic_coopt", np.nan))
    v2g_val = float(summary.get("v2g_only", np.nan))
    gap = stoch_val - v2g_val if np.isfinite(stoch_val) and np.isfinite(v2g_val) else np.nan

    # Primary chart
    fig = go.Figure()
    for strat in summary.index:
        fig.add_trace(go.Bar(
            x=[STRATEGY_LABELS.get(strat, strat)], y=[summary[strat]],
            marker_color=STRATEGY_COLORS[strat], showlegend=False,
            text=[f"${summary[strat]:,.0f}"], textposition="outside",
        ))
    fig.update_layout(
        yaxis_title="$ per vehicle per year", height=400,
        title=dict(text=f"Annual revenue at {multiplier:.2f}× inference price",
                   font=dict(size=14, color=TEXT), x=0),
    )
    st.plotly_chart(themed(fig), use_container_width=True)

    # Inline summary
    stat_row([
        f'<span class="accent">${stoch_val:,.0f}</span> stochastic co-opt',
        f'<strong>${v2g_val:,.0f}</strong> V2G-only',
        f'<strong>{"+" if gap >= 0 else ""}${gap:,.0f}</strong> advantage',
    ])

    callout(
        "<strong>Defensive proof.</strong> At multiplier = 0 the inference market has disappeared. "
        "The platform still beats passive and smart-charging-only baselines — it degrades "
        "gracefully instead of failing. Not a one-bet product."
    )


def tab_comparison() -> None:
    tab_lead(
        "Five strategies, same one-year fleet trace. Stochastic co-optimization is the only "
        "one that plans across all three revenue streams under forecast uncertainty."
    )

    summary = load_backtest_summary()
    if summary is None:
        missing_backtest()
        return

    summary = summary.reindex([s for s in STRATEGY_ORDER if s in summary.index])
    labels = [STRATEGY_LABELS.get(s, s) for s in summary.index]

    # Primary chart: revenue
    fig = go.Figure()
    for strat in summary.index:
        fig.add_trace(go.Bar(
            x=[STRATEGY_LABELS.get(strat, strat)],
            y=[summary.loc[strat, "total_revenue_usd"]],
            marker_color=STRATEGY_COLORS[strat], showlegend=False,
            text=[f"${summary.loc[strat, 'total_revenue_usd']:,.0f}"],
            textposition="outside",
        ))
    fig.update_layout(
        yaxis_title="$ per vehicle per year", height=400,
        title=dict(text="Annual revenue per vehicle",
                   font=dict(size=14, color=TEXT), x=0),
    )
    st.plotly_chart(themed(fig), use_container_width=True)

    # Secondary: decomposition (full width)
    section_label("REVENUE DECOMPOSITION")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="V2G", x=labels, y=summary["v2g_revenue_usd"],
                         marker_color="#94e2d5"))   # teal
    fig.add_trace(go.Bar(name="Inference", x=labels, y=summary["inference_revenue_usd"],
                         marker_color="#fab387"))   # peach
    fig.add_trace(go.Bar(name="Charging cost (−)", x=labels,
                         y=-summary["charging_cost_usd"], marker_color=TEXT_DIM))
    fig.update_layout(
        barmode="relative", yaxis_title="$ per vehicle per year", height=320,
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    st.plotly_chart(themed(fig), use_container_width=True)

    # Tertiary: reliability + MWh (2-up)
    c1, c2 = st.columns(2)
    with c1:
        rel = summary["mobility_reliability"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=rel,
            marker_color=[STRATEGY_COLORS[s] for s in summary.index],
            text=[f"{r:.1f}%" for r in rel], textposition="outside", showlegend=False,
        ))
        fig.add_hline(y=99, line=dict(color=ACCENT, dash="dash"),
                      annotation_text="99%", annotation=dict(font_color=ACCENT_DIM))
        fig.update_layout(
            yaxis=dict(title="% hours", range=[max(0, rel.min() - 2), 100.5]),
            height=300,
            title=dict(text="Mobility reliability",
                       font=dict(size=14, color=TEXT), x=0),
        )
        st.plotly_chart(themed(fig), use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=summary["mwh_in_events"],
            marker_color=[STRATEGY_COLORS[s] for s in summary.index],
            text=[f"{m:.2f}" for m in summary["mwh_in_events"]],
            textposition="outside", showlegend=False,
        ))
        fig.update_layout(
            yaxis_title="MWh / vehicle / year", height=300,
            title=dict(text="MWh delivered during grid events",
                       font=dict(size=14, color=TEXT), x=0),
        )
        st.plotly_chart(themed(fig), use_container_width=True)

    # Full table — under an expander so it doesn't clutter the main view
    with st.expander("Full metrics table"):
        display = summary.copy()
        display.index = labels
        display["mobility_reliability"] = display["mobility_reliability"] * 100
        display = display.rename(columns={
            "total_revenue_usd": "Total rev ($)",
            "v2g_revenue_usd": "V2G rev ($)",
            "inference_revenue_usd": "Infer rev ($)",
            "charging_cost_usd": "Charging ($)",
            "total_charge_kwh": "Charge (kWh)",
            "total_discharge_kwh": "Discharge (kWh)",
            "infer_hours": "Infer (h)",
            "mwh_in_events": "MWh in events",
            "mobility_reliability": "Reliability (%)",
        })
        st.dataframe(display.round(2), use_container_width=True)


# ---------- Sidebar ----------

def render_sidebar(fleet: pd.DataFrame, grid: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.1rem; font-weight:700; color:{TEXT}; margin-bottom:0.2rem;">'
            'OpenVPP Orchestrator</div>'
            f'<div style="color:{TEXT_MUTED}; font-size:0.8rem; margin-bottom:1.8rem;">'
            'Software For Energy · Arizona</div>',
            unsafe_allow_html=True,
        )

        n_events = int(grid["event_id"].dropna().nunique())
        st.markdown(
            f"""
            <div style="color:{TEXT}; font-size:0.86rem; line-height:1.85; margin-bottom:1.8rem;">
              <b>{len(fleet)}</b> vehicles (70P / 30F)<br>
              <b>{n_events}</b> grid events / year<br>
              24h rolling-horizon MILP<br>
              Mobility risk priced, not walled
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="color:{TEXT_MUTED}; font-size:0.78rem; letter-spacing:0.04em; margin-bottom:0.35rem;">REPRODUCE</div>
            <div style="font-size:0.82rem; color:{TEXT}; line-height:1.75;">
              <code>make install</code><br>
              <code>make reproduce</code><br>
              <code>make dash</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="position:absolute; bottom:1.5rem; color:{TEXT_DIM}; font-size:0.75rem;">'
            'Solo build · 48 hours</div>',
            unsafe_allow_html=True,
        )


# ---------- Main ----------

def main() -> None:
    st.set_page_config(
        page_title="OpenVPP Orchestrator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    fleet = load_fleet()
    lmp = load_lmp()
    grid = load_grid_events()
    summary = load_backtest_summary()

    render_sidebar(fleet, grid)

    # Header — title + one-line subtitle
    st.markdown(
        '<div class="brand-title">OpenVPP Orchestrator</div>'
        '<div class="brand-sub">One battery, three revenue streams. '
        'Co-optimized every hour, under uncertainty.</div>',
        unsafe_allow_html=True,
    )

    # Hero — single big number
    render_hero(summary, fleet, grid)

    # Tabs
    t1, t2, t3, t4 = st.tabs([
        "Live fleet",
        "Grid event simulator",
        "Economics sensitivity",
        "Baseline comparison",
    ])
    with t1:
        tab_live_fleet(fleet, lmp, grid)
    with t2:
        tab_event_simulator(grid, lmp)
    with t3:
        tab_sensitivity()
    with t4:
        tab_comparison()


if __name__ == "__main__":
    main()
