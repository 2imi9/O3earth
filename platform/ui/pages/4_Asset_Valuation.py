"""Asset Valuation — NPV/IRR/LCOE with full 25-year cash flow charts."""

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
    valuation_kpi_row,
    cash_flow_chart,
    revenue_cost_chart,
    generation_chart,
    price_projection_chart,
)

init_state()
render_sidebar()

st.header("Asset Valuation")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("valuation_form"):
    st.subheader("Asset Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        asset_id = st.text_input("Asset ID", value="asset_001")
        asset_types = [
            "solar_utility", "solar_distributed", "wind_onshore",
            "wind_offshore", "hydro", "battery_storage",
        ]
        idx = asset_types.index(st.session_state.asset_type) if st.session_state.asset_type in asset_types else 0
        asset_type = st.selectbox("Asset Type", asset_types, index=idx)
        US_STATES = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        ]
        state_idx = US_STATES.index(st.session_state.selected_state) if st.session_state.selected_state in US_STATES else 4
        state = st.selectbox("State", US_STATES, index=state_idx)

    with col2:
        latitude = st.number_input("Latitude", value=st.session_state.selected_lat, step=0.01)
        longitude = st.number_input("Longitude", value=st.session_state.selected_lon, step=0.01)
        capacity_mw = st.number_input("Capacity (MW)", value=st.session_state.capacity_mw, min_value=0.1, step=10.0)
        capacity_factor = st.slider("Capacity Factor", 0.05, 0.60, 0.25, 0.01)

    with col3:
        installation_cost = st.number_input("Install Cost ($/kW)", value=1000.0, step=50.0)
        fixed_om = st.number_input("Fixed O&M ($/kW/yr)", value=15.0, step=1.0)
        degradation = st.number_input("Degradation Rate", value=0.005, step=0.001, format="%.3f")
        project_life = st.number_input("Project Life (years)", value=25, min_value=5, max_value=40)

    st.subheader("Verification & Discount")
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        verification_status = st.selectbox("Verification Status", ["verified", "pending", "flagged"])
    with vcol2:
        verification_confidence = st.slider("Verification Confidence", 0.0, 1.0, 0.0, 0.01)
    with vcol3:
        discount_rate = st.slider("Discount Rate", 0.04, 0.20, 0.08, 0.01)

    submitted = st.form_submit_button("Calculate Valuation", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if submitted:
    api = get_api_client()
    with st.spinner("Calculating valuation..."):
        result = api.value_asset(
            asset_id=asset_id,
            asset_type=asset_type,
            latitude=latitude,
            longitude=longitude,
            state=state,
            capacity_mw=capacity_mw,
            capacity_factor=capacity_factor,
            installation_cost_per_kw=installation_cost,
            fixed_om_per_kw_year=fixed_om,
            degradation_rate=degradation,
            project_life_years=project_life,
            verification_status=verification_status,
            verification_confidence=verification_confidence,
            discount_rate=discount_rate,
        )

    if result:
        st.session_state.last_valuation = result

        st.divider()
        st.subheader("Valuation Results")

        # KPI row
        valuation_kpi_row(result)

        # Extra metrics
        extra1, extra2, extra3 = st.columns(3)
        extra1.metric("Verification-Adj NPV", f"${result['verification_adjusted_npv']:,.0f}")
        extra2.metric("VaR (95%)", f"${result['value_at_risk_95']:,.0f}")
        extra3.metric("Verification Discount", f"{result['verification_discount']:.1%}")

        st.divider()

        # Charts
        st.plotly_chart(cash_flow_chart(result), use_container_width=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(revenue_cost_chart(result), use_container_width=True)
        with chart_right:
            st.plotly_chart(generation_chart(result), use_container_width=True)

        st.plotly_chart(price_projection_chart(result), use_container_width=True)

        # Assumptions
        with st.expander("Assumptions"):
            st.json(result.get("assumptions", {}))

        # Tokenization
        with st.expander("Tokenization Metrics"):
            st.markdown(
                "Tokenization metrics would be calculated from this valuation. "
                "The API provides a `/api/tokenize` endpoint for this."
            )
            st.metric("Risk-Adj NPV (tokenizable)", f"${result['risk_adjusted_npv']:,.0f}")

elif st.session_state.last_valuation:
    result = st.session_state.last_valuation
    st.info(f"Showing previous valuation for asset: {result['asset_id']}")
    valuation_kpi_row(result)
    st.plotly_chart(cash_flow_chart(result), use_container_width=True)
