"""Site Selection — pick a site, compare Factor Engine vs ML suitability."""

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
st.markdown("Click on the map to select a site, then score its suitability.")

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

st.markdown(
    f"**Current selection:** {st.session_state.selected_lat:.4f}, "
    f"{st.session_state.selected_lon:.4f}"
)

st.divider()

# ------------------------------------------------------------------
# Score
# ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Score Suitability", use_container_width=True, type="primary"):
        api = get_api_client()
        results = {}
        with st.spinner("Scoring energy types..."):
            for etype in ["solar", "wind"]:
                r = api.score_suitability(
                    latitude=st.session_state.selected_lat,
                    longitude=st.session_state.selected_lon,
                    energy_type=etype,
                )
                if r:
                    results[etype] = r

        if results:
            st.session_state.last_suitability_all = results

with col2:
    st.page_link("pages/0_AI_Chat.py", label="Ask AI", icon="💬", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if st.session_state.get("last_suitability_all"):
    results = st.session_state.last_suitability_all
    icons = {"solar": "☀️", "wind": "💨", "hydro": "💧"}

    # ------------------------------------------------------------------
    # Factor Engine scores
    # ------------------------------------------------------------------
    st.subheader("Factor Engine")
    st.caption("Rule-based scoring from real-time APIs (NASA POWER, USGS, Open-Meteo)")

    fcols = st.columns(2)
    for i, etype in enumerate(["solar", "wind"]):
        if etype in results:
            score = results[etype].get("factor_score", 0)
            fcols[i].metric(f"{icons[etype]} {etype.title()}", f"{score:.0%}")

    # ------------------------------------------------------------------
    # OlmoEarth ML scores
    # ------------------------------------------------------------------
    st.subheader("OlmoEarth ML")
    st.caption("Satellite embedding similarity to known energy sites (XGBoost on 768-dim OlmoEarth features)")

    mcols = st.columns(2)
    for i, etype in enumerate(["solar", "wind"]):
        if etype in results:
            r = results[etype]
            if r.get("ml_available") and r.get("ml_score") is not None:
                mcols[i].metric(f"{icons[etype]} {etype.title()}", f"{r['ml_score']:.0%}")
            else:
                mcols[i].metric(f"{icons[etype]} {etype.title()}", "—")

    # ------------------------------------------------------------------
    # Factor Breakdown
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Factor Breakdown")

    for etype in ["solar", "wind"]:
        if etype not in results:
            continue
        r = results[etype]
        factors = r.get("factors", {})
        data_sources = r.get("data_sources", {})
        if not factors:
            continue

        with st.expander(f"{icons.get(etype, '')} {etype.title()}"):
            real = {k: v for k, v in factors.items() if v != 0.5}
            estimated = {k: v for k, v in factors.items() if v == 0.5}

            if real:
                num_cols = min(len(real), 4)
                ecols = st.columns(num_cols)
                for j, (name, value) in enumerate(real.items()):
                    ecols[j % num_cols].metric(name, f"{value:.0%}")

            if estimated:
                st.caption(f"⚠️ {len(estimated)} factors estimated: {', '.join(estimated.keys())}")

            if data_sources:
                with st.expander("Data sources"):
                    for key, info in data_sources.items():
                        st.text(f"{key}: {info['value']} ({info['source']})")
