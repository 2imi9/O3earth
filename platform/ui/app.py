"""OpenEnergy Engine — Streamlit entry point."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from ui.*` imports work
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(_root) / ".env")
except ImportError:
    pass  # In Docker, env vars come from docker-compose

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar

st.set_page_config(
    page_title="OpenEnergy Engine",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
render_sidebar()

# Landing page
st.title("OpenEnergy Engine")
st.markdown(
    """
    **AI-powered Earth observation for renewable energy verification and NEMS-based valuation.**

    Use the sidebar to pick a location, then navigate to the pages below.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Site Selection")
    st.write("Interactive map to pick renewable energy sites. Click anywhere on the map to select coordinates.")
    st.page_link("pages/2_Site_Selection.py", label="Open Map", icon="🗺️")

with col2:
    st.subheader("Climate Risk")
    st.write("Assess climate risk under SSP scenarios. View extreme event probabilities and resource projections.")
    st.page_link("pages/3_Climate_Risk.py", label="Assess Risk", icon="🌡️")

with col3:
    st.subheader("Asset Valuation")
    st.write("Calculate NPV, IRR, LCOE with 25-year cash flow projections and tokenization metrics.")
    st.page_link("pages/4_Asset_Valuation.py", label="Value Asset", icon="📈")

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Detection")
    st.write("Detect solar and wind installations from satellite imagery using the OlmoEarth vision transformer.")
    st.page_link("pages/5_Detection.py", label="Run Detection", icon="🛰️")

with col5:
    st.subheader("AI Chat")
    st.write("Chat with NVIDIA NIM LLM. Ask questions, analyze results, generate reports.")
    st.page_link("pages/6_AI_Chat.py", label="Open Chat", icon="💬")

with col6:
    st.subheader("Dashboard")
    st.write("Overview of your most recent analyses, API health, and module availability.")
    st.page_link("pages/1_Dashboard.py", label="View Dashboard", icon="📊")
