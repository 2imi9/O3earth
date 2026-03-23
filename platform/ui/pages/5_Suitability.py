"""Suitability — site suitability scoring using OlmoEarth embeddings."""

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
st.markdown("Score locations for renewable energy suitability using OlmoEarth foundation model embeddings + configurable factor engine.")

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
    with st.spinner(f"Computing {energy_type} suitability score..."):
        result = api.score_suitability(
            latitude=latitude,
            longitude=longitude,
            energy_type=energy_type,
        )

    if result:
        st.session_state.last_suitability = result

        st.divider()
        st.subheader("Suitability Results")

        combined = result.get("combined_score", 0.0)
        factor_score = result.get("factor_score", 0.0)
        ml_score = result.get("ml_score")
        ml_available = result.get("ml_available", False)
        confidence = result.get("confidence", "unknown")

        if confidence == "high":
            st.success(f"**High suitability** for {energy_type}: {combined:.1%}")
        elif confidence == "moderate":
            st.warning(f"**Moderate suitability** for {energy_type}: {combined:.1%}")
        else:
            st.error(f"**Low suitability** for {energy_type}: {combined:.1%}")

        # Score comparison
        col1, col2, col3 = st.columns(3)
        col1.metric("Combined Score", f"{combined:.1%}")
        col2.metric("Factor Engine", f"{factor_score:.1%}")
        if ml_available:
            col3.metric("ML Model (OlmoEarth)", f"{ml_score:.1%}")
        else:
            col3.metric("ML Model", "No nearby embedding")

        st.caption("Combined = 60% ML + 40% Factor Engine (when ML available)")

        # Factor breakdown
        factors = result.get("factors", {})
        if factors:
            st.divider()
            st.subheader("Factor Breakdown")
            cols = st.columns(min(len(factors), 4))
            for i, (name, value) in enumerate(factors.items()):
                cols[i % len(cols)].metric(name.replace("_", " ").title(), f"{value:.2f}")

        # Next steps
        st.divider()
        st.markdown("**Next Steps**")
        st.page_link("pages/3_Climate_Risk.py", label="Assess Climate Risk for this site")
        st.page_link("pages/0_AI_Chat.py", label="Ask AI about this location")

    else:
        st.error("Could not compute suitability score. Check API connection.")

elif st.session_state.get("last_suitability"):
    result = st.session_state.last_suitability
    st.info("Showing previous result. Submit a new query above.")
    score = result.get("suitability_score", 0)
    st.metric("Last Score", f"{score:.1%}")
