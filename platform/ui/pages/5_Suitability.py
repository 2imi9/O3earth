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
st.markdown("Score locations for renewable energy suitability using OlmoEarth foundation model embeddings + XGBoost classifier.")

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
        date_range = st.text_input("Imagery Date Range", value="2022-01-01/2023-12-31")

    submitted = st.form_submit_button("Score Site", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if submitted:
    api = get_api_client()
    with st.spinner("Fetching satellite imagery and computing suitability score..."):
        # For now, call the detection endpoint — will be replaced with suitability endpoint
        result = api.detect(
            latitude=latitude,
            longitude=longitude,
            date_range=date_range,
            max_cloud_cover=20.0,
        )

    if result:
        st.session_state.last_detection = result

        st.divider()
        st.subheader("Suitability Results")

        # Score display
        score = result.get("detection_confidence", 0.0)
        if score > 0.7:
            st.success(f"**High suitability** for {energy_type}: {score:.1%}")
        elif score > 0.4:
            st.warning(f"**Moderate suitability** for {energy_type}: {score:.1%}")
        else:
            st.error(f"**Low suitability** for {energy_type}: {score:.1%}")

        # Detail metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Suitability Score", f"{score:.1%}")
        col2.metric("Energy Type", energy_type.title())
        col3.metric("Location", f"{latitude:.3f}, {longitude:.3f}")

        # Next steps
        st.divider()
        st.markdown("**Next Steps**")
        st.page_link("pages/3_Climate_Risk.py", label="Assess Climate Risk for this site")
        st.page_link("pages/4_Asset_Valuation.py", label="Value this asset")
        st.page_link("pages/0_AI_Chat.py", label="Ask AI about this location")

    else:
        st.error("Could not compute suitability score. Check API connection.")

elif st.session_state.get("last_detection"):
    st.info("Showing previous result. Submit a new query above.")
