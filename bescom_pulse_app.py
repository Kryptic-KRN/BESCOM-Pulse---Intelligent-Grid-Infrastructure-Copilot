
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BESCOM Pulse — Grid Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stSidebar"] { background-color: #111827; }
.block-container { padding-top: 1rem; }
.metric-card { background: #1a2235; border: 1px solid rgba(99,179,237,0.2);
               border-radius: 12px; padding: 16px; margin-bottom: 8px; }
.status-green  { color: #10b981; font-weight: 700; }
.status-yellow { color: #f59e0b; font-weight: 700; }
.status-orange { color: #f97316; font-weight: 700; }
.status-red    { color: #ef4444; font-weight: 700; }
.tier-badge    { display:inline-block; padding:2px 8px; border-radius:4px;
                 font-size:12px; font-weight:600; margin-right:4px; }
</style>""", unsafe_allow_html=True)

# ─── Load Data & Engine ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from pathlib import Path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "bescom_pulse_masked_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please ensure the CSV exists next to the script.")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df

df = load_data()

STATUS_COLOR = {"GREEN": "#10b981", "YELLOW": "#f59e0b", "ORANGE": "#f97316", "RED": "#ef4444"}

# Zone summary
@st.cache_data
def build_zone_summary(df):
    zs = (
        df.groupby(["zone_id","zone_name","zone_lat","zone_lon","demand_class"])
        .agg(
            avg_load_pct=("load_pct","mean"),
            max_load_pct=("load_pct","max"),
            avg_capacity_mw=("zone_capacity_mw","mean"),
            total_sessions=("record_id","count"),
            tier4_events=("infra_expansion_flag","sum"),
            anomaly_count=("anomaly_flag","sum"),
            avg_peak_rate=("peak_rate_inr_kwh","mean"),
            avg_offpeak_rate=("offpeak_rate_inr_kwh","mean"),
        ).reset_index().round(2)
    )
    zs["status"] = zs["avg_load_pct"].apply(
        lambda x: "GREEN" if x < 70 else ("YELLOW" if x < 85 else ("ORANGE" if x < 95 else "RED"))
    )
    return zs

zone_df = build_zone_summary(df)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ BESCOM Pulse")
    st.markdown("*Intelligent Grid & Infrastructure Copilot*")
    st.divider()
    page = st.radio("Navigate", [
        "📊 Overview",
        "🗺️ Zone Heatmap",
        "⚙️ Optimization Engine",
        "💬 MCP Chat Interface",
        "🏗️ Infrastructure Planning",
        "📈 Demand Analytics",
    ])
    st.divider()
    # Live stats in sidebar
    red_count   = (zone_df.status == "RED").sum()
    orange_count = (zone_df.status == "ORANGE").sum()
    st.markdown(f"🔴 **{red_count}** Critical Zones")
    st.markdown(f"🟠 **{orange_count}** Warning Zones")
    st.markdown(f"📅 Data: {df.timestamp.max().strftime('%d %b %Y')}")
    st.markdown("🔒 *Privacy: All data masked*")

# ─── Helper Functions ─────────────────────────────────────────────────────────────
def status_badge(s):
    colors = {"GREEN":"🟢","YELLOW":"🟡","ORANGE":"🟠","RED":"🔴"}
    return f"{colors.get(s,"⚪")} {s}"

def run_optimization_logic(zone_id):
    """Standalone optimization runner (no engine class dependency)."""
    row = zone_df[zone_df.zone_id == zone_id].iloc[0]
    load = row.avg_load_pct
    results = {}
    # Tier 1 — always
    results["tier1"] = {
        "triggered": True,
        "message": f"Recommend 60% SoC cap for all sessions in {row.zone_name}. Reduces charging window by ~40% and extends battery life.",
        "soc_cap": 60,
    }
    # Tier 2
    if load >= 70:
        savings = round((row.avg_peak_rate - row.avg_offpeak_rate) / row.avg_peak_rate * 100, 1)
        results["tier2"] = {
            "triggered": True,
            "message": (
                f"{row.zone_name} at {load:.1f}% load. "
                f"Charge NOW at ₹{row.avg_peak_rate:.2f}/kWh OR shift to off-peak "
                f"at ₹{row.avg_offpeak_rate:.2f}/kWh. Save {savings}%."
            ),
        }
    # Tier 3
    if load >= 85:
        near = zone_df[zone_df.avg_load_pct < 70].nsmallest(1,"avg_load_pct")
        if not near.empty:
            n = near.iloc[0]
            results["tier3"] = {
                "triggered": True,
                "message": f"Redirect to {n.zone_name} ({n.avg_load_pct:.1f}% load — GREEN zone).",
            }
    # Tier 4
    if load >= 95:
        infra = df[(df.zone_id == zone_id) & (df.infra_expansion_flag == 1)]
        nlat = infra.suggested_nodal_lat.mean() if not infra.empty else row.zone_lat + 0.01
        nlon = infra.suggested_nodal_lon.mean() if not infra.empty else row.zone_lon + 0.01
        results["tier4"] = {
            "triggered": True,
            "message": (
                f"CRITICAL: {row.zone_name} needs new charging hub at "
                f"({nlat:.4f}°N, {nlon:.4f}°E). "
                f"Est. CAPEX: ₹{int(row.tier4_events * 8.5 + 120)}L"
            ),
            "nodal_lat": nlat,
            "nodal_lon": nlon,
        }
    return load, results


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("⚡ BESCOM Pulse — Grid Overview")
    st.caption(f"Last updated: {df.timestamp.max().strftime('%d %b %Y, %H:%M')} | 🔒 All data masked & anonymised")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sessions", f"{len(df):,}", "30-day window")
    c2.metric("Zones Monitored", zone_df.zone_id.nunique())
    c3.metric("🔴 Critical Zones", red_count, delta=f"+{red_count} need attention", delta_color="inverse")
    c4.metric("Infra Triggers", int(df.infra_expansion_flag.sum()))
    c5.metric("Anomalies Detected", int(df.anomaly_flag.sum()))

    st.divider()

    # Zone table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Zone Status Dashboard")
        display = zone_df[["zone_name","avg_load_pct","max_load_pct","avg_capacity_mw",
                            "total_sessions","tier4_events","status"]].copy()
        display.columns = ["Zone","Avg Load %","Peak Load %","Capacity (MW)",
                            "Sessions","Infra Triggers","Status"]
        def color_status(val):
            c = {"GREEN":"background-color:#0a2e1f;color:#10b981",
                 "YELLOW":"background-color:#2e2200;color:#f59e0b",
                 "ORANGE":"background-color:#2e1500;color:#f97316",
                 "RED":"background-color:#2e0a0a;color:#ef4444"}
            return c.get(val, "")
        # Streamlit sometimes doesn't accept pandas Styler across environments.
        # Pre-format the DataFrame instead and render it directly.
        disp_sorted = display.sort_values("Avg Load %", ascending=False).copy()
        # Format numeric columns as strings for consistent display
        disp_sorted["Avg Load %"] = disp_sorted["Avg Load %"].map(lambda x: f"{x:.1f}%")
        disp_sorted["Peak Load %"] = disp_sorted["Peak Load %"].map(lambda x: f"{x:.1f}%")
        disp_sorted["Capacity (MW)"] = disp_sorted["Capacity (MW)"].map(lambda x: f"{x:.1f}")
        # Replace status with a badge/emoji for easy reading
        disp_sorted["Status"] = disp_sorted["Status"].map(lambda s: status_badge(s))
        st.dataframe(disp_sorted, height=400, use_container_width=True)

    with col2:
        st.subheader("Status Distribution")
        sc = zone_df.status.value_counts().reset_index()
        sc.columns = ["Status","Count"]
        fig = px.pie(sc, values="Count", names="Status",
                     color="Status",
                     color_discrete_map=STATUS_COLOR,
                     hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=280, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tier Activations")
        tc = df.tier_triggered.value_counts().sort_index().reset_index()
        tc.columns = ["Tier","Count"]
        tc["Tier"] = tc["Tier"].map({1:"T1: SoC Cap",2:"T2: Time-Shift",3:"T3: Redirect",4:"T4: Infra"})
        tier_colors = ["#10b981","#f59e0b","#3b82f6","#ef4444"]
        fig2 = px.bar(tc, x="Tier", y="Count", color="Tier",
                      color_discrete_sequence=tier_colors)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", showlegend=False, height=220,
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ZONE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Zone Heatmap":
    st.title("🗺️ Bengaluru Zone Heatmap")
    st.caption("Interactive map of all 15 zones — colour = current load status")

    fig = px.scatter_mapbox(
        zone_df,
        lat="zone_lat", lon="zone_lon",
        color="avg_load_pct",
        size="total_sessions",
        hover_name="zone_name",
        hover_data={"avg_load_pct":":.1f","max_load_pct":":.1f",
                    "tier4_events":True,"status":True},
        color_continuous_scale=[[0,"#10b981"],[0.7,"#f59e0b"],[0.85,"#f97316"],[1,"#ef4444"]],
        range_color=[40, 105],
        zoom=10.5,
        center={"lat": 12.95, "lon": 77.62},
        mapbox_style="carto-darkmatter",
        title="Zone Load % (size = session volume)",
        height=560,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                      margin=dict(t=40,b=0,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

    # Infra nodes
    infra_data = df[df.infra_expansion_flag == 1].dropna(subset=["suggested_nodal_lat"])
    if not infra_data.empty:
        st.subheader("🏗️ Proposed Infra Expansion Nodes")
        infra_agg = infra_data.groupby("zone_id").agg(
            zone_name=("zone_name","first"),
            nodal_lat=("suggested_nodal_lat","mean"),
            nodal_lon=("suggested_nodal_lon","mean"),
            events=("record_id","count")
        ).reset_index()
        fig2 = px.scatter_mapbox(
            infra_agg, lat="nodal_lat", lon="nodal_lon",
            hover_name="zone_name", size="events",
            color_discrete_sequence=["#a78bfa"],
            zoom=10.5, center={"lat":12.95,"lon":77.62},
            mapbox_style="carto-darkmatter", height=400,
            title="Optimal Nodal Points for New Charging Hubs"
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",
                           margin=dict(t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Optimization Engine":
    st.title("⚙️ 4-Tier Optimization Engine")

    # Tier flow explainer
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("**🟢 Tier 1: SoC Cap**")
        st.caption("Always active. Cap charging at 60% SoC. Extends battery life, reduces load window.")
    with t2:
        st.markdown("**🟡 Tier 2: Time-Shift**")
        st.caption("Load > 70%. Incentive nudge — charge off-peak for lower ₹/kWh.")
    with t3:
        st.markdown("**🟠 Tier 3: Spatial Redirect**")
        st.caption("Load > 85%. Route EV to nearest Green Zone charging station.")
    with t4:
        st.markdown("**🔴 Tier 4: Infra Trigger**")
        st.caption("Load > 95%. Flag zone for new hub. Compute optimal nodal point (x,y).")

    st.divider()

    zone_options = {r.zone_name: r.zone_id for _, r in zone_df.iterrows()}
    selected_zone = st.selectbox("Select Zone to Analyse", list(zone_options.keys()), index=0)
    zone_id = zone_options[selected_zone]

    row = zone_df[zone_df.zone_id == zone_id].iloc[0]
    load, tier_results = run_optimization_logic(zone_id)

    # Load gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=load,
        number={"suffix": "%", "font":{"size":40}},
        delta={"reference": 70, "increasing":{"color":"#ef4444"}, "decreasing":{"color":"#10b981"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#64748b"},
            "bar":{"color": STATUS_COLOR.get(row.status, "#3b82f6")},
            "steps":[
                {"range":[0,70],  "color":"rgba(16,185,129,0.15)"},
                {"range":[70,85], "color":"rgba(245,158,11,0.15)"},
                {"range":[85,95], "color":"rgba(249,115,22,0.15)"},
                {"range":[95,100],"color":"rgba(239,68,68,0.15)"},
            ],
            "threshold":{"line":{"color":"#ef4444","width":4},"thickness":0.75,"value":95},
        },
        title={"text": f"{selected_zone} — Current Load", "font":{"size":16}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=280,
                      margin=dict(t=40,b=20,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)

    # Tier results
    st.subheader(f"Tier Recommendations for {selected_zone}")
    tier_labels = {
        "tier1": ("🟢", "Tier 1: SoC Cap"),
        "tier2": ("🟡", "Tier 2: Time-Shift Incentive"),
        "tier3": ("🟠", "Tier 3: Spatial Redirect"),
        "tier4": ("🔴", "Tier 4: Infrastructure Trigger"),
    }
    for key, (icon, label) in tier_labels.items():
        if key in tier_results:
            with st.expander(f"{icon} {label} — TRIGGERED", expanded=(key in ["tier3","tier4"])):
                st.info(tier_results[key]["message"])
                if key == "tier1":
                    st.progress(0.60, text="Recommended SoC Cap: 60%")
                if key == "tier2" and load >= 70:
                    c1, c2 = st.columns(2)
                    c1.metric("Peak Rate", f"₹{row.avg_peak_rate:.2f}/kWh", "Charge now")
                    c2.metric("Off-Peak Rate", f"₹{row.avg_offpeak_rate:.2f}/kWh", "⬇ Better")
                if key == "tier4" and "nodal_lat" in tier_results["tier4"]:
                    st.success(f"📍 Optimal Hub Location: "
                               f"{tier_results['tier4']['nodal_lat']:.4f}°N, "
                               f"{tier_results['tier4']['nodal_lon']:.4f}°E")
        else:
            with st.expander(f"⚪ {tier_labels[key][1]} — Not triggered", expanded=False):
                st.caption("Thresholds not met.")

    # Approve / Edit / Override
    st.divider()
    st.subheader("Operator Decision")
    cc1, cc2, cc3 = st.columns(3)
    if cc1.button("✅ Approve Recommendations", use_container_width=True, type="primary"):
        st.success(f"Recommendations approved for {selected_zone}. Actions queued.")
    if cc2.button("✏️ Edit Before Applying", use_container_width=True):
        st.info("Edit mode: Override individual tier actions below.")
    if cc3.button("🚫 Override — No Action", use_container_width=True):
        st.warning(f"Manual override: no automated actions for {selected_zone}.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MCP CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 MCP Chat Interface":
    st.title("💬 MCP Chat — BESCOM Pulse Copilot")
    st.caption("Ask anything about zone load, optimization actions, or infrastructure planning.")

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": (
                "👋 Welcome to BESCOM Pulse. I have real-time access to grid telemetry across all 15 zones in Bengaluru. "
                "Ask me about zone status, optimization recommendations, or infrastructure planning.\n\n"
                "Try: *'Why is HSR Layout showing a red alert?'* or *'Which zones need new charging hubs?'*"
            )}
        ]
    if "mcp_history" not in st.session_state:
        st.session_state.mcp_history = []

    # Rule-based fallback responses (no API key needed)
    def fallback_response(query: str) -> str:
        q = query.lower()

        # Zone status queries
        for _, row in zone_df.iterrows():
            if row.zone_name.lower() in q:
                load, tiers = run_optimization_logic(row.zone_id)
                msgs = [t["message"] for t in tiers.values()]
                return (
                    f"**{row.zone_name}** is at **{load:.1f}% load** — Status: {status_badge(row.status)}\n\n"
                    + "\n".join(f"• {m}" for m in msgs)
                )

        if any(w in q for w in ["red", "critical", "overload", "worst"]):
            worst = zone_df.nlargest(3, "avg_load_pct")
            lines = [f"**Top 3 critical zones:**"]
            for _, r in worst.iterrows():
                lines.append(f"• {r.zone_name}: {r.avg_load_pct:.1f}% ({r.status})")
            return "\n".join(lines)

        if any(w in q for w in ["infra", "hub", "expand", "new", "node", "capex"]):
            infra = zone_df[zone_df.tier4_events > 0].nlargest(3, "tier4_events")
            lines = ["**Infrastructure expansion recommended:**"]
            for _, r in infra.iterrows():
                lines.append(
                    f"• {r.zone_name}: {int(r.tier4_events)} overload events — Est. CAPEX ₹{int(r.tier4_events*8.5+120)}L"
                )
            return "\n".join(lines)

        if any(w in q for w in ["summary", "overview", "all zones", "status"]):
            lines = [f"**All Zone Status:**"]
            for _, r in zone_df.sort_values("avg_load_pct", ascending=False).iterrows():
                lines.append(f"• {r.zone_name}: {r.avg_load_pct:.1f}% — {status_badge(r.status)}")
            return "\n".join(lines)

        if any(w in q for w in ["soc", "battery", "60", "cap"]):
            return (
                "**60% SoC Cap Strategy (Tier 1)**\n\n"
                "Recommending a 60% SoC cap achieves two goals:\n\n"
                "• **Grid**: Reduces the charging window by ~40%, freeing capacity for other users\n\n"
                "• **Battery**: 15-20% longer cycle life — most EV batteries degrade faster above 80%\n\n"
                "We frame this as *'charge smart, not full'* to drive adoption."
            )

        if any(w in q for w in ["anomal", "unusual", "alert"]):
            anom = zone_df.nlargest(3, "anomaly_count")
            lines = ["**Anomaly Report:**"]
            for _, r in anom.iterrows():
                lines.append(f"• {r.zone_name}: {int(r.anomaly_count)} anomalies detected")
            return "\n".join(lines)

        return (
            f"I have access to grid data for all 15 Bengaluru zones. "
            f"Currently {red_count} zones are in RED status and {int(df.infra_expansion_flag.sum())} "
            f"infrastructure triggers have been logged. Ask me about a specific zone or topic!"
        )

    # Chat UI
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggestion chips
    st.markdown("**Quick queries:**")
    chips = [
        "Why is HSR Layout showing a red alert?",
        "Which zones need infrastructure expansion?",
        "Show me all zone status",
        "Explain the 60% SoC cap strategy",
        "Which zones have anomalies?",
    ]
    cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if cols[i].button(chip[:30]+".." if len(chip)>30 else chip, key=f"chip_{i}"):
            st.session_state._chip_query = chip

    # Handle chip clicks
    chip_query = st.session_state.pop("_chip_query", None)

    user_input = st.chat_input("Ask anything about the grid...")
    query = chip_query or user_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Querying MCP server..."):
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
                if api_key:
                    # Live AI response would go here — needs anthropic package
                    response = fallback_response(query)
                else:
                    response = fallback_response(query)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = st.session_state.chat_history[:1]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INFRASTRUCTURE PLANNING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Infrastructure Planning":
    st.title("🏗️ Infrastructure Planning — Nodal Strategy")
    st.caption("Identifies optimal locations for new EV charging hubs based on demand center-of-mass.")

    infra = df[df.infra_expansion_flag == 1].dropna(subset=["suggested_nodal_lat"])
    infra_agg = infra.groupby(["zone_id","zone_name"]).agg(
        events=("record_id","count"),
        nodal_lat=("suggested_nodal_lat","mean"),
        nodal_lon=("suggested_nodal_lon","mean"),
        avg_load=("load_pct","mean"),
        avg_ev_kw=("ev_load_contribution_kw","mean"),
    ).reset_index().round(4)
    infra_agg["capex_lakhs"] = (infra_agg["events"] * 8.5 + 120).astype(int)
    infra_agg["priority"] = infra_agg["events"].apply(
        lambda x: "🔴 High" if x >= 20 else ("🟡 Medium" if x >= 10 else "🟢 Low")
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Expansion Zones")
        st.dataframe(
            infra_agg[["zone_name","events","avg_load","nodal_lat","nodal_lon","capex_lakhs","priority"]]
            .sort_values("events", ascending=False)
            .rename(columns={"zone_name":"Zone","events":"Overload Events",
                              "avg_load":"Avg Load %","nodal_lat":"Hub Lat",
                              "nodal_lon":"Hub Lon","capex_lakhs":"CAPEX (₹L)","priority":"Priority"}),
            use_container_width=True, height=350
        )

    with col2:
        st.subheader("CAPEX by Zone")
        fig = px.bar(
            infra_agg.sort_values("capex_lakhs"),
            x="capex_lakhs", y="zone_name",
            orientation="h",
            color="avg_load",
            color_continuous_scale="RdYlGn_r",
            labels={"capex_lakhs":"CAPEX (₹ Lakhs)","zone_name":"Zone"},
            height=350
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0",margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Nodal Map — Optimal Hub Locations")
    fig3 = go.Figure()
    # Existing zones
    fig3.add_trace(go.Scattermapbox(
        lat=zone_df.zone_lat, lon=zone_df.zone_lon,
        mode="markers",
        marker=dict(
            size=12,
            color=[STATUS_COLOR.get(s,"#3b82f6") for s in zone_df.status],
            opacity=0.7
        ),
        text=zone_df.zone_name,
        name="Existing Zones"
    ))
    # Proposed nodes
    fig3.add_trace(go.Scattermapbox(
        lat=infra_agg.nodal_lat, lon=infra_agg.nodal_lon,
        mode="markers+text",
        marker=dict(size=18, color="#a78bfa", symbol="star"),
        text=infra_agg.zone_name,
        textposition="top center",
        name="Proposed Hubs"
    ))
    fig3.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=12.95,lon=77.62), zoom=10.5),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        height=500, margin=dict(t=0,b=0,l=0,r=0),
        legend=dict(bgcolor="rgba(17,24,39,0.8)")
    )
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Analytics":
    st.title("📈 Demand Analytics")

    tab1, tab2, tab3 = st.tabs(["Hourly Demand", "Charger Mix", "SoC & Pricing"])

    with tab1:
        st.subheader("Average Load % by Hour of Day")
        hourly = df.groupby("hour_of_day").agg(
            avg_load=("load_pct","mean"),
            avg_ev_kw=("ev_load_contribution_kw","mean"),
            sessions=("record_id","count")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly.hour_of_day, y=hourly.avg_load,
                             name="Avg Load %",
                             marker_color=["#ef4444" if (8<=h<=10 or 17<=h<=21) else "#3b82f6"
                                           for h in hourly.hour_of_day]))
        fig.add_hline(y=70, line_dash="dash", line_color="#f59e0b", annotation_text="T2 Threshold")
        fig.add_hline(y=85, line_dash="dash", line_color="#f97316", annotation_text="T3 Threshold")
        fig.add_hline(y=95, line_dash="dash", line_color="#ef4444", annotation_text="T4 Threshold")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0",xaxis_title="Hour",yaxis_title="Load %",height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red bars = peak hours (8–10am, 5–9pm)")

    with tab2:
        st.subheader("Charger Type Distribution")
        c1, c2 = st.columns(2)
        with c1:
            charger_dist = df.charger_type.value_counts().reset_index()
            charger_dist.columns = ["Type","Count"]
            fig = px.pie(charger_dist, values="Count", names="Type", hole=0.4,
                         color_discrete_sequence=["#10b981","#3b82f6","#f59e0b","#ef4444"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            avg_load_by_charger = df.groupby("charger_type").agg(
                avg_load=("load_pct","mean"),
                avg_ev_kw=("ev_load_contribution_kw","mean")
            ).reset_index()
            fig2 = px.bar(avg_load_by_charger, x="charger_type", y="avg_ev_kw",
                          color="avg_load", color_continuous_scale="RdYlGn_r",
                          labels={"charger_type":"Charger","avg_ev_kw":"Avg EV Load (kW)"},
                          height=300)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("SoC Distribution at Session Start")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="soc_initial_pct", nbins=20,
                               color_discrete_sequence=["#3b82f6"],
                               labels={"soc_initial_pct":"SoC at Session Start (%)"},
                               height=280)
            fig.add_vline(x=60, line_dash="dash", line_color="#10b981",
                          annotation_text="60% Cap")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Peak vs Off-Peak Pricing")
            pricing = df.groupby("is_peak_hour").agg(
                avg_peak=("peak_rate_inr_kwh","mean"),
                avg_off=("offpeak_rate_inr_kwh","mean")
            ).reset_index()
            pricing["period"] = pricing.is_peak_hour.map({0:"Off-Peak",1:"Peak"})
            fig2 = go.Figure(data=[
                go.Bar(name="Peak Rate", x=pricing.period, y=pricing.avg_peak,
                       marker_color="#ef4444"),
                go.Bar(name="Off-Peak Rate", x=pricing.period, y=pricing.avg_off,
                       marker_color="#10b981"),
            ])
            fig2.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",
                               yaxis_title="₹/kWh",height=280)
            st.plotly_chart(fig2, use_container_width=True)
