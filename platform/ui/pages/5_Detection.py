"""Detection — satellite-based renewable energy detection."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar
from ui.utils.api_client import get_api_client

init_state()
render_sidebar()

st.header("Renewable Energy Detection")
st.markdown("Detect solar and wind installations from satellite imagery using the OlmoEarth vision transformer.")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("detection_form"):
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=st.session_state.selected_lat, step=0.01)
        longitude = st.number_input("Longitude", value=st.session_state.selected_lon, step=0.01)

    with col2:
        date_range = st.text_input("Date Range (YYYY-MM-DD/YYYY-MM-DD)", value="2024-01-01/2024-12-31")
        max_cloud_cover = st.slider("Max Cloud Cover (%)", 0, 100, 10)

    submitted = st.form_submit_button("Run Detection", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if submitted:
    api = get_api_client()
    with st.spinner("Running satellite detection model..."):
        result = api.detect(
            latitude=latitude,
            longitude=longitude,
            date_range=date_range,
            max_cloud_cover=float(max_cloud_cover),
        )

    if result:
        st.session_state.last_detection = result

        st.divider()
        st.subheader("Detection Results")

        # Main result banner
        if result["detected"]:
            st.success(f"Installation DETECTED with {result['detection_confidence']:.1%} confidence")
        else:
            st.warning("No installation detected at this location.")

        # Detail metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detected", "Yes" if result["detected"] else "No")
        col2.metric("Confidence", f"{result['detection_confidence']:.1%}")
        col3.metric("Classification", result["classification"])
        col4.metric("Class Confidence", f"{result['classification_confidence']:.1%}")

        st.divider()

        col5, col6, col7 = st.columns(3)
        col5.metric("Estimated Capacity", f"{result['estimated_capacity_mw']:.1f} MW")
        col6.metric("Image Date", result["image_date"])
        col7.metric("Image Source", result["image_source"])

        # Quick links
        if result["detected"]:
            st.divider()
            st.markdown("**Next Steps**")
            st.page_link("pages/3_Climate_Risk.py", label="Assess Climate Risk for this site")
            st.page_link("pages/4_Asset_Valuation.py", label="Value this asset")

elif st.session_state.last_detection:
    result = st.session_state.last_detection
    st.info("Showing previous detection result.")
    if result["detected"]:
        st.success(f"Installation detected: {result['classification']} ({result['detection_confidence']:.1%})")
    else:
        st.warning("No installation detected.")
    st.metric("Estimated Capacity", f"{result['estimated_capacity_mw']:.1f} MW")
