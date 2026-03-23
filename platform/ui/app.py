"""O3 EartH — Streamlit entry point."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from ui.*` imports work
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
# Also add project root for `src.*` imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(_root) / ".env")
except ImportError:
    pass

import streamlit as st
from ui.utils.state import init_state
from ui.components.sidebar import render_sidebar

st.set_page_config(
    page_title="O3 EartH",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
render_sidebar()

# Landing page — redirect to AI Chat as default
st.title("O3 EartH")
st.markdown(
    """
    **Geospatial site suitability assessment using foundation model embeddings.**

    Use the sidebar to navigate, or start with AI Chat below.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("AI Chat")
    st.write("Chat with NVIDIA NIM LLM. Ask questions about site suitability, analyze locations, generate reports.")
    st.page_link("pages/0_AI_Chat.py", label="Open Chat", icon="💬")

with col2:
    st.subheader("Site Selection")
    st.write("Interactive map to select locations for suitability analysis.")
    st.page_link("pages/2_Site_Selection.py", label="Open Map", icon="🗺️")

with col3:
    st.subheader("Suitability Scoring")
    st.write("Score locations for solar, wind, hydro, or geothermal suitability using OlmoEarth embeddings.")
    st.page_link("pages/5_Suitability.py", label="Score Site", icon="🛰️")

st.divider()

col4, col5 = st.columns(2)

with col4:
    st.subheader("Climate Risk")
    st.write("Assess climate risk under SSP scenarios. View extreme event probabilities and resource projections.")
    st.page_link("pages/3_Climate_Risk.py", label="Assess Risk", icon="🌡️")

with col5:
    st.subheader("Dashboard")
    st.write("Overview of recent analyses, API health, and module availability.")
    st.page_link("pages/1_Dashboard.py", label="View Dashboard", icon="📊")
