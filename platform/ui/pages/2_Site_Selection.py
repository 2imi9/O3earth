"""Site Selection — score a location for a chosen energy type."""

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
st.markdown("Select an energy type, pick a location, and see how it scores.")

# ------------------------------------------------------------------
# Energy type selector
# ------------------------------------------------------------------
energy_type = st.radio(
    "Energy Type",
    ["solar", "wind"],
    format_func=lambda x: {"solar": "☀️ Solar", "wind": "💨 Wind"}[x],
    horizontal=True,
)

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

st.markdown(
    f"**Location:** {st.session_state.selected_lat:.4f}, "
    f"{st.session_state.selected_lon:.4f}"
)

# ------------------------------------------------------------------
# Score this location
# ------------------------------------------------------------------
if st.button("Score This Location", use_container_width=True, type="primary"):
    api = get_api_client()
    with st.spinner(f"Scoring for {energy_type}..."):
        r = api.score_suitability(
            latitude=st.session_state.selected_lat,
            longitude=st.session_state.selected_lon,
            energy_type=energy_type,
        )
    if r:
        st.session_state.last_score = r
        st.session_state.last_energy_type = energy_type

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if st.session_state.get("last_score"):
    r = st.session_state.last_score
    etype = st.session_state.get("last_energy_type", "solar")
    icon = {"solar": "☀️", "wind": "💨"}.get(etype, "")

    st.divider()

    # Two scoring methods side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Factor Engine")
        st.metric(f"{icon} {etype.title()}", f"{r.get('factor_score', 0):.0%}")
        st.caption("Real-time data from NASA POWER, Open-Elevation, etc.")

    with col2:
        st.subheader("OlmoEarth ML")
        if r.get("ml_available") and r.get("ml_score") is not None:
            st.metric(f"{icon} {etype.title()}", f"{r['ml_score']:.0%}")
            st.caption("Satellite embedding similarity to known sites")
        else:
            st.metric(f"{icon} {etype.title()}", "—")
            st.caption("No pre-computed embedding near this location")

    # Interpretation
    factor = r.get("factor_score", 0)
    ml = r.get("ml_score")
    ml_avail = r.get("ml_available", False)

    st.divider()
    if factor > 0.7:
        st.success(f"This location has strong {etype} potential based on measured conditions.")
    elif factor > 0.4:
        st.warning(f"This location has moderate {etype} potential. Some factors are favorable.")
    else:
        st.error(f"This location has weak {etype} potential based on current data.")

    if ml_avail and ml is not None:
        if ml > 0.8:
            st.info(f"OlmoEarth confirms: this area resembles existing {etype} installations.")
        elif ml < 0.3:
            st.info(f"OlmoEarth suggests: this area does not resemble typical {etype} sites.")

    # Factor breakdown
    factors = r.get("factors", {})
    data_sources = r.get("data_sources", {})

    if factors:
        with st.expander("Factor Breakdown"):
            real = {k: v for k, v in factors.items() if v != 0.5}
            estimated = {k: v for k, v in factors.items() if v == 0.5}

            if real:
                num_cols = min(len(real), 4)
                fcols = st.columns(num_cols)
                for j, (name, value) in enumerate(real.items()):
                    fcols[j % num_cols].metric(name, f"{value:.0%}")

            if estimated:
                st.caption(f"⚠️ {len(estimated)} factors estimated: {', '.join(estimated.keys())}")

        if data_sources:
            with st.expander("Data Sources"):
                for key, info in data_sources.items():
                    st.text(f"{key}: {info['value']} ({info['source']})")

    # ------------------------------------------------------------------
    # Compare with other locations
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Compare Locations")
    st.caption(f"How does this site compare to other {etype} locations?")

    # Score all presets for the same energy type
    if st.button("Compare with preset locations"):
        from ui.components.sidebar import PRESETS
        api = get_api_client()
        comparisons = []

        current_lat = st.session_state.selected_lat
        current_lon = st.session_state.selected_lon

        with st.spinner("Scoring preset locations..."):
            # Current location
            comparisons.append({
                "Location": f"📍 Current ({current_lat:.2f}, {current_lon:.2f})",
                "Factor": f"{r.get('factor_score', 0):.0%}",
                "ML": f"{r['ml_score']:.0%}" if r.get('ml_available') and r.get('ml_score') is not None else "—",
            })

            for name, (lat, lon, _) in PRESETS.items():
                if name == "Custom" or lat is None:
                    continue
                pr = api.score_suitability(latitude=lat, longitude=lon, energy_type=etype)
                if pr:
                    comparisons.append({
                        "Location": name,
                        "Factor": f"{pr.get('factor_score', 0):.0%}",
                        "ML": f"{pr['ml_score']:.0%}" if pr.get('ml_available') and pr.get('ml_score') is not None else "—",
                    })

        if comparisons:
            import pandas as pd
            st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)
