"""Site Selection — pick a site, compare Factor Engine vs ML suitability."""

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
# Score
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

with col2:
    st.page_link("pages/0_AI_Chat.py", label="Ask AI", icon="💬", use_container_width=True)

# ------------------------------------------------------------------
# Results — Dual Score Comparison
# ------------------------------------------------------------------
if st.session_state.get("last_suitability_all"):
    results = st.session_state.last_suitability_all
    icons = {"solar": "☀️", "wind": "💨", "hydro": "💧"}

    # Best type by factor score
    best_factor = max(results, key=lambda t: results[t].get("factor_score", 0))

    # Check if any ML scores available
    any_ml = any(r.get("ml_available", False) for r in results.values())

    st.subheader("Comparison: Factor Engine vs OlmoEarth ML")

    if any_ml:
        st.caption("Two independent scoring methods — agreement = high confidence")
    else:
        st.caption("ML scores available when location is near a pre-computed embedding")

    # Header row
    header_cols = st.columns([2, 1.5, 1.5, 1])
    header_cols[0].markdown("**Energy Type**")
    header_cols[1].markdown("**Factor Engine** *(real-time APIs)*")
    header_cols[2].markdown("**OlmoEarth ML** *(satellite embeddings)*")
    header_cols[3].markdown("**Agreement**")

    st.divider()

    for etype in ["solar", "wind", "hydro"]:
        if etype not in results:
            continue
        r = results[etype]
        factor = r.get("factor_score", 0)
        ml = r.get("ml_score")
        ml_avail = r.get("ml_available", False)
        is_best = etype == best_factor

        row = st.columns([2, 1.5, 1.5, 1])

        # Energy type label
        label = f"{icons.get(etype, '')} **{etype.title()}**"
        if is_best:
            label += " 🏆"
        row[0].markdown(label)

        # Factor engine score
        row[1].metric("Factor", f"{factor:.0%}", label_visibility="collapsed")

        # ML score
        if ml_avail and ml is not None:
            row[2].metric("ML", f"{ml:.0%}", label_visibility="collapsed")
        else:
            row[2].markdown("*No nearby embedding*")

        # Agreement indicator
        if ml_avail and ml is not None:
            diff = abs(factor - ml)
            if diff < 0.15:
                row[3].markdown("✅ Agree")
            elif diff < 0.30:
                row[3].markdown("⚠️ Differ")
            else:
                row[3].markdown("❌ Disagree")
        else:
            row[3].markdown("—")

    # ------------------------------------------------------------------
    # Factor Breakdown
    # ------------------------------------------------------------------
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

        is_best = etype == best_factor
        with st.expander(
            f"{icons.get(etype, '')} {etype.title()} — Factor: {r.get('factor_score', 0):.0%}"
            + (f" | ML: {r['ml_score']:.0%}" if r.get('ml_available') and r.get('ml_score') is not None else ""),
            expanded=is_best,
        ):
            real = {k: v for k, v in factors.items() if v != 0.5}
            estimated = {k: v for k, v in factors.items() if v == 0.5}

            if real:
                fcols = st.columns(min(len(real), 4))
                for j, (name, value) in enumerate(real.items()):
                    fcols[j % len(fcols)].metric(name, f"{value:.0%}")

            if estimated:
                st.caption(f"⚠️ {len(estimated)} factors estimated: {', '.join(estimated.keys())}")

            if data_sources:
                with st.expander("Data sources"):
                    for key, info in data_sources.items():
                        st.text(f"{key}: {info['value']} ({info['source']})")
