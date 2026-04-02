"""Plotly chart helpers for the dashboard."""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List


def valuation_kpi_row(v: dict):
    """Render a row of KPI metrics for a valuation result."""
    cols = st.columns(5)
    cols[0].metric("NPV", f"${v['npv_usd']:,.0f}")
    cols[1].metric("IRR", f"{v['irr'] * 100:.1f}%")
    cols[2].metric("LCOE", f"${v['lcoe_per_mwh']:.2f}/MWh")
    cols[3].metric("Payback", f"{v['payback_years']:.1f} yr")
    cols[4].metric("Risk-Adj NPV", f"${v['risk_adjusted_npv']:,.0f}")


def cash_flow_chart(v: dict) -> go.Figure:
    """Bar + line chart of annual net cash flows and cumulative."""
    years = list(range(1, len(v["annual_net_cash_flow"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=years,
            y=v["annual_net_cash_flow"],
            name="Net Cash Flow",
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=v["cumulative_cash_flow"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="orange", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Cash Flow Projection",
        xaxis_title="Year",
        yaxis_title="Annual Net ($)",
        yaxis2=dict(title="Cumulative ($)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        height=400,
    )
    return fig


def revenue_cost_chart(v: dict) -> go.Figure:
    """Stacked bar of revenue vs costs."""
    years = list(range(1, len(v["annual_revenue_usd"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=v["annual_revenue_usd"], name="Revenue", marker_color="green"))
    fig.add_trace(go.Bar(x=years, y=v["annual_costs_usd"], name="Costs", marker_color="tomato"))
    fig.update_layout(
        title="Revenue vs Costs",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="USD",
        height=350,
    )
    return fig


def generation_chart(v: dict) -> go.Figure:
    """Annual generation with degradation."""
    years = list(range(1, len(v["annual_generation_mwh"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=v["annual_generation_mwh"],
            mode="lines+markers",
            name="Generation (MWh)",
            line=dict(color="teal"),
        )
    )
    fig.update_layout(
        title="Annual Generation (with degradation)",
        xaxis_title="Year",
        yaxis_title="MWh",
        height=350,
    )
    return fig


def price_projection_chart(v: dict) -> go.Figure:
    """Electricity price projection."""
    years = list(range(1, len(v["electricity_prices"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=v["electricity_prices"],
            mode="lines+markers",
            name="$/MWh",
            line=dict(color="purple"),
        )
    )
    fig.update_layout(
        title="Electricity Price Projection",
        xaxis_title="Year",
        yaxis_title="$/MWh",
        height=350,
    )
    return fig


def climate_risk_radar(risk: dict) -> go.Figure:
    """Radar chart of extreme event probabilities."""
    probs = risk.get("extreme_event_probs", {})
    if not probs:
        return go.Figure()

    labels = list(probs.keys())
    values = list(probs.values())
    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="Event Probability",
            line=dict(color="crimson"),
        )
    )
    fig.update_layout(
        title="Extreme Event Probabilities",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
    )
    return fig


def climate_resource_bar(risk: dict) -> go.Figure:
    """Bar chart for solar GHI and wind speed P10/P50/P90."""
    fig = go.Figure()

    ghi = risk.get("solar_ghi_kwh_m2_year", {})
    wind = risk.get("wind_speed_m_s", {})

    percentiles = ["p10", "p50", "p90"]

    if ghi:
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=[ghi.get(p, 0) for p in percentiles],
                name="Solar GHI (kWh/m2/yr)",
                marker_color="gold",
            )
        )

    if wind:
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=[wind.get(p, 0) for p in percentiles],
                name="Wind Speed (m/s)",
                marker_color="skyblue",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title="Resource Assessment (P10 / P50 / P90)",
        barmode="group",
        yaxis_title="Solar GHI (kWh/m2/yr)",
        yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right"),
        height=400,
    )
    return fig


def risk_score_gauge(score: float) -> go.Figure:
    """Gauge chart for overall risk score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Risk Score"},
            gauge=dict(
                axis=dict(range=[0, 1]),
                bar=dict(color="darkblue"),
                steps=[
                    dict(range=[0, 0.3], color="lightgreen"),
                    dict(range=[0.3, 0.6], color="gold"),
                    dict(range=[0.6, 1.0], color="salmon"),
                ],
            ),
        )
    )
    fig.update_layout(height=250)
    return fig
