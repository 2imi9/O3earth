"""Shared sidebar component."""

import streamlit as st
from ui.utils.api_client import get_api_client
from ui.utils.state import init_state


# Preset locations for quick selection
PRESETS = {
    "Custom": (None, None, ""),
    "Alabama, US (Solar)": (30.84, -87.34, "AL"),
    "Kansas, US (Wind)": (37.55, -98.24, "KS"),
    "Wisconsin, US (Hydro)": (44.79, -89.69, "WI"),
    "Geysers, CA (Geothermal)": (38.75, -122.72, "CA"),
    "Poland (Solar)": (51.94, 22.89, ""),
    "Italy (Wind)": (36.80, 14.55, ""),
    "Germany (Hydro)": (53.38, 11.59, ""),
    "Turkey (Geothermal)": (38.44, 28.39, ""),
}


def render_sidebar():
    """Render the shared sidebar with location picker and status."""
    init_state()

    st.sidebar.title("O3 EartH")
    st.sidebar.caption("Geospatial Site Suitability Assessment")
    st.sidebar.divider()

    # API status
    api = get_api_client()
    health = api.health()
    if health:
        st.sidebar.success("API Connected")
        modules = health.get("modules", {})
        available = [k for k, v in modules.items() if v]
        st.sidebar.caption(f"Modules: {', '.join(available) if available else 'core only'}")
    else:
        st.sidebar.error("API Disconnected")
        st.sidebar.caption("Start the API: `uvicorn api.main:app`")

    st.sidebar.divider()

    # Quick location selector
    st.sidebar.subheader("Quick Location")
    preset = st.sidebar.selectbox("Preset Sites", list(PRESETS.keys()))
    if preset != "Custom":
        lat, lon, state = PRESETS[preset]
        st.session_state.selected_lat = lat
        st.session_state.selected_lon = lon
        if state:
            st.session_state.selected_state = state

    st.sidebar.number_input(
        "Latitude",
        value=st.session_state.selected_lat,
        min_value=-90.0,
        max_value=90.0,
        step=0.01,
        key="sidebar_lat",
        on_change=lambda: setattr(st.session_state, "selected_lat", st.session_state.sidebar_lat),
    )
    st.sidebar.number_input(
        "Longitude",
        value=st.session_state.selected_lon,
        min_value=-180.0,
        max_value=180.0,
        step=0.01,
        key="sidebar_lon",
        on_change=lambda: setattr(st.session_state, "selected_lon", st.session_state.sidebar_lon),
    )

    st.sidebar.divider()
    st.sidebar.caption("Northeastern University")
