"""Site Selection — pick a site, score suitability across all energy types."""

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
# Score all 4 energy types
# ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Score Suitability", use_container_width=True, type="primary"):
        api = get_api_client()
        results = {}
        with st.spinner("Scoring energy types..."):
            for etype in ["solar", "wind", "hydro"]:
                r = api.score_suitability(
                    latitude=st.session_state.selected_lat,
                    longitude=st.session_state.selected_lon,
                    energy_type=etype,
                )
                if r:
                    results[etype] = r

        if results:
            st.session_state.last_suitability_all = results

            # Find best type
            best_type = max(results, key=lambda t: results[t].get("combined_score", 0))
            best_score = results[best_type].get("combined_score", 0)
            best_confidence = results[best_type].get("confidence", "unknown")

            st.session_state.last_suitability = results[best_type]

with col2:
    st.page_link("pages/0_AI_Chat.py", label="Ask AI", icon="💬", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if st.session_state.get("last_suitability_all"):
    results = st.session_state.last_suitability_all

    best_type = max(results, key=lambda t: results[t].get("combined_score", 0))
    best_score = results[best_type].get("combined_score", 0)

    st.subheader(f"Best: {best_type.title()} ({best_score:.0%})")

    # Show 3 types (geothermal excluded — insufficient real-time data for demo)
    cols = st.columns(3)
    icons = {"solar": "☀️", "wind": "💨", "hydro": "💧"}

    for i, etype in enumerate(["solar", "wind", "hydro"]):
        with cols[i]:
            if etype in results:
                score = results[etype].get("combined_score", 0)
                is_best = etype == best_type
                label = f"{icons.get(etype, '')} {etype.title()}"
                if is_best:
                    st.metric(label, f"{score:.0%}", "BEST")
                else:
                    st.metric(label, f"{score:.0%}")
            else:
                st.metric(f"{icons.get(etype, '')} {etype.title()}", "N/A")

    # Factor breakdown for each type
    st.divider()
    st.subheader("Factor Breakdown")

    for etype in ["solar", "wind", "hydro"]:
        if etype not in results:
            continue
        r = results[etype]
        factors = r.get("factors", {})
        data_sources = r.get("data_sources", {})
        if not factors:
            continue

        with st.expander(f"{icons.get(etype, '')} {etype.title()} — {r.get('combined_score', 0):.0%}", expanded=(etype == best_type)):
            real = {k: v for k, v in factors.items() if v != 0.5}
            estimated = {k: v for k, v in factors.items() if v == 0.5}

            if real:
                fcols = st.columns(min(len(real), 4))
                for j, (name, value) in enumerate(real.items()):
                    fcols[j % len(fcols)].metric(name, f"{value:.0%}")

            if estimated:
                st.caption(f"⚠️ {len(estimated)} factors estimated (no real-time data): {', '.join(estimated.keys())}")

            if data_sources:
                with st.expander("Data sources"):
                    for key, info in data_sources.items():
                        st.text(f"{key}: {info['value']} ({info['source']})")
