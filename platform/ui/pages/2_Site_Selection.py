"""Site Selection — interactive map for picking renewable energy sites."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from ui.utils.state import init_state, update_location
from ui.components.sidebar import render_sidebar
from ui.components.map_widget import render_site_map
from ui.utils.api_client import get_api_client

init_state()
render_sidebar()

st.header("Site Selection")
st.markdown("Click on the map to select a site for analysis. The red marker shows the current selection.")

# ------------------------------------------------------------------
# Map
# ------------------------------------------------------------------
clicked_lat, clicked_lon = render_site_map(
    center_lat=st.session_state.selected_lat,
    center_lon=st.session_state.selected_lon,
    zoom=6,
    key="site_select_map",
)

if clicked_lat is not None:
    update_location(clicked_lat, clicked_lon)
    st.success(f"Selected: ({clicked_lat:.4f}, {clicked_lon:.4f})")

# Show current coordinates
st.markdown(
    f"**Current selection:** {st.session_state.selected_lat:.4f}, "
    f"{st.session_state.selected_lon:.4f} "
    f"(State: {st.session_state.selected_state})"
)

st.divider()

# ------------------------------------------------------------------
# Quick-analyze buttons
# ------------------------------------------------------------------
st.subheader("Analyze This Site")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Run Climate Risk", use_container_width=True):
        api = get_api_client()
        with st.spinner("Assessing climate risk..."):
            result = api.assess_climate_risk(
                latitude=st.session_state.selected_lat,
                longitude=st.session_state.selected_lon,
                elevation=st.session_state.selected_elevation,
            )
        if result:
            st.session_state.last_climate_risk = result
            st.success(f"Risk Score: {result['risk_score']:.3f}")
        else:
            st.error("Climate risk assessment failed.")

with col2:
    if st.button("Score Suitability", use_container_width=True):
        api = get_api_client()
        with st.spinner("Computing suitability score..."):
            result = api.score_suitability(
                latitude=st.session_state.selected_lat,
                longitude=st.session_state.selected_lon,
                energy_type=st.session_state.get("energy_type", "solar"),
            )
        if result:
            st.session_state.last_suitability = result
            score = result.get("combined_score", result.get("suitability_score", 0))
            confidence = result.get("confidence", "unknown")
            if confidence == "high":
                st.success(f"High suitability: {score:.1%}")
            elif confidence == "moderate":
                st.warning(f"Moderate suitability: {score:.1%}")
            else:
                st.info(f"Low suitability: {score:.1%}")
        else:
            st.error("Suitability scoring failed.")

with col3:
    st.page_link("pages/0_AI_Chat.py", label="Ask AI about this site", icon="💬", use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# State selector
# ------------------------------------------------------------------
st.subheader("Site Parameters")
col_a, col_b, col_c = st.columns(3)

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

with col_a:
    idx = US_STATES.index(st.session_state.selected_state) if st.session_state.selected_state in US_STATES else 4
    st.session_state.selected_state = st.selectbox("State", US_STATES, index=idx)

with col_b:
    st.session_state.selected_elevation = st.number_input("Elevation (m)", value=st.session_state.selected_elevation, step=10.0)

with col_c:
    energy_types = ["solar", "wind", "hydro", "geothermal"]
    st.session_state.energy_type = st.selectbox("Energy Type", energy_types, index=0)
