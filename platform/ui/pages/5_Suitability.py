"""Suitability — site suitability scoring."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar
from ui.utils.api_client import get_api_client

init_state()
render_sidebar()

st.header("Site Suitability Scoring")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("suitability_form"):
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=st.session_state.selected_lat, step=0.01)
        longitude = st.number_input("Longitude", value=st.session_state.selected_lon, step=0.01)

    with col2:
        energy_type = st.selectbox("Energy Type", ["solar", "wind", "hydro", "geothermal"])

    submitted = st.form_submit_button("Score Site", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if submitted:
    api = get_api_client()
    with st.spinner(f"Computing {energy_type} suitability..."):
        result = api.score_suitability(
            latitude=latitude,
            longitude=longitude,
            energy_type=energy_type,
        )

    if result:
        st.session_state.last_suitability = result

        st.divider()

        score = result.get("combined_score", 0.0)
        confidence = result.get("confidence", "unknown")

        # Main score
        if confidence == "high":
            st.success(f"**High suitability** for {energy_type}: {score:.0%}")
        elif confidence == "moderate":
            st.warning(f"**Moderate suitability** for {energy_type}: {score:.0%}")
        else:
            st.error(f"**Low suitability** for {energy_type}: {score:.0%}")

        # Factor breakdown
        factors = result.get("factors", {})
        if factors:
            st.subheader("Factor Breakdown")
            # Show only factors that aren't default 0.5
            active = {k: v for k, v in factors.items() if v != 0.5}
            default = {k: v for k, v in factors.items() if v == 0.5}

            if active:
                cols = st.columns(min(len(active), 4))
                for i, (name, value) in enumerate(active.items()):
                    cols[i % len(cols)].metric(name, f"{value:.0%}")

            if default:
                with st.expander(f"{len(default)} factors pending data connection"):
                    for name in default:
                        st.caption(f"- {name}")

        # Next steps
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/3_Climate_Risk.py", label="Assess Climate Risk", icon="🌡️")
        with col2:
            st.page_link("pages/0_AI_Chat.py", label="Ask AI about this site", icon="💬")

    else:
        st.error("Could not compute suitability score. Check API connection.")
