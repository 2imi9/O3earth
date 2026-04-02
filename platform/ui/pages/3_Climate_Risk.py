"""Climate Risk Assessment — risk scoring, extreme events, resource projections."""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar
from ui.utils.api_client import get_api_client
from ui.components.charts import (
    climate_risk_radar,
    climate_resource_bar,
    risk_score_gauge,
)

init_state()
render_sidebar()

st.header("Climate Risk Assessment")
st.caption("Current climate data from NASA POWER. Projections based on IPCC AR6 SSP scenarios.")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("climate_risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=st.session_state.selected_lat, step=0.01)
        longitude = st.number_input("Longitude", value=st.session_state.selected_lon, step=0.01)
        elevation = st.number_input("Elevation (m)", value=st.session_state.selected_elevation, step=10.0)

    with col2:
        asset_type = st.selectbox("Asset Type", ["solar", "wind"])
        scenario = st.selectbox("Climate Scenario", ["SSP126", "SSP245", "SSP370", "SSP585"], index=1)
        target_year = st.slider("Target Year", 2025, 2100, st.session_state.target_year)

    submitted = st.form_submit_button("Assess Risk", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if submitted:
    api = get_api_client()
    with st.spinner("Running climate risk model..."):
        result = api.assess_climate_risk(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            asset_type=asset_type,
            scenario=scenario,
            target_year=target_year,
        )

    if result:
        st.session_state.last_climate_risk = result

        st.divider()
        st.subheader("Results")

        # KPI row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Risk Score", f"{result['risk_score']:.3f}")
        kpi2.metric("Solar GHI (P50)", f"{result['solar_ghi_kwh_m2_year']['p50']:.0f} kWh/m2/yr")
        kpi3.metric("Wind Speed (P50)", f"{result['wind_speed_m_s']['p50']:.1f} m/s")
        kpi4.metric("Temp Change", f"{result['temperature_change_c']:+.1f} C")

        st.divider()

        # Charts
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.plotly_chart(risk_score_gauge(result["risk_score"]), use_container_width=True)
            st.plotly_chart(climate_risk_radar(result), use_container_width=True)

        with chart_right:
            st.plotly_chart(climate_resource_bar(result), use_container_width=True)

            # Additional metrics
            st.markdown("**Additional Details**")
            st.metric("Precipitation Change", f"{result['precipitation_change_pct']:+.1f}%")
            st.metric("Confidence", f"{result['confidence']:.0%}")

        # Scenario comparison
        st.divider()
        st.subheader("Scenario Comparison")
        st.markdown("Run all four SSP scenarios to compare risk levels.")

        if st.button("Compare All Scenarios"):
            scenarios = ["SSP126", "SSP245", "SSP370", "SSP585"]
            results = {}
            progress = st.progress(0)
            for i, scen in enumerate(scenarios):
                r = api.assess_climate_risk(
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    asset_type=asset_type,
                    scenario=scen,
                    target_year=target_year,
                )
                if r:
                    results[scen] = r
                progress.progress((i + 1) / len(scenarios))

            if results:
                comp_cols = st.columns(len(results))
                for col, (scen, r) in zip(comp_cols, results.items()):
                    col.metric(f"{scen} Risk", f"{r['risk_score']:.3f}")
                    col.metric(f"{scen} Temp", f"{r['temperature_change_c']:+.1f} C")
                    col.metric(f"{scen} GHI P50", f"{r['solar_ghi_kwh_m2_year']['p50']:.0f}")

elif st.session_state.last_climate_risk:
    # Show previous results
    result = st.session_state.last_climate_risk
    st.info(f"Showing previous result for ({result['latitude']:.2f}, {result['longitude']:.2f})")
    st.metric("Risk Score", f"{result['risk_score']:.3f}")
    st.plotly_chart(climate_risk_radar(result), use_container_width=True)
