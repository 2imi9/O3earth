"""Dashboard — overview, health status, recent analyses."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar
from ui.utils.api_client import get_api_client
from ui.components.charts import valuation_kpi_row, climate_risk_radar

init_state()
render_sidebar()

st.header("Dashboard")

# ------------------------------------------------------------------
# API health & module availability
# ------------------------------------------------------------------
api = get_api_client()
health = api.health()

if health:
    modules = health.get("modules", {})
    st.subheader("Module Availability")
    cols = st.columns(len(modules))
    for col, (name, available) in zip(cols, modules.items()):
        col.metric(
            name.capitalize(),
            "Available" if available else "Unavailable",
        )
else:
    st.warning("Cannot reach the API backend. Start it with: `uvicorn api.main:app --reload`")

st.divider()

# ------------------------------------------------------------------
# Last analysis summaries
# ------------------------------------------------------------------
st.subheader("Recent Analyses")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Last Valuation**")
    if st.session_state.last_valuation:
        v = st.session_state.last_valuation
        valuation_kpi_row(v)
    else:
        st.info("No valuation run yet. Go to Asset Valuation to start.")

with col_right:
    st.markdown("**Last Climate Risk**")
    if st.session_state.last_climate_risk:
        risk = st.session_state.last_climate_risk
        st.metric("Risk Score", f"{risk['risk_score']:.3f}")
        st.metric("Solar GHI (P50)", f"{risk['solar_ghi_kwh_m2_year']['p50']:.0f} kWh/m2/yr")
        st.metric("Wind Speed (P50)", f"{risk['wind_speed_m_s']['p50']:.1f} m/s")
    else:
        st.info("No climate risk assessment yet. Go to Climate Risk to start.")

st.divider()

# ------------------------------------------------------------------
# Last detection
# ------------------------------------------------------------------
st.markdown("**Last Detection**")
if st.session_state.last_detection:
    d = st.session_state.last_detection
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("Detected", "Yes" if d["detected"] else "No")
    dcol2.metric("Classification", d["classification"])
    dcol3.metric("Capacity", f"{d['estimated_capacity_mw']:.1f} MW")
else:
    st.info("No detection run yet. Go to Detection to start.")
