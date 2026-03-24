"""AI Chat — ask questions about site suitability and energy data."""

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

st.header("AI Chat")

api = get_api_client()

# Check LLM status
llm_info = api.llm_status()
if llm_info and llm_info.get("available"):
    st.caption("Powered by NVIDIA NIM")
else:
    st.warning("No LLM available. Add `NVIDIA_API_KEY` to `.env`")

# ------------------------------------------------------------------
# Suggested prompts (show only when chat is empty)
# ------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_response" not in st.session_state:
    st.session_state.pending_response = False

if not st.session_state.chat_messages:
    col1, col2 = st.columns(2)
    if col1.button("What can O3 EartH do?", use_container_width=True):
        st.session_state.chat_messages.append({"role": "user", "content": "What can O3 EartH do?"})
        st.session_state.pending_response = True
        st.rerun()
    if col2.button("How does OlmoEarth scoring work?", use_container_width=True):
        st.session_state.chat_messages.append({"role": "user", "content": "How does OlmoEarth scoring work?"})
        st.session_state.pending_response = True
        st.rerun()

# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input from chat box
prompt = st.chat_input("Ask about site suitability, energy data, climate risk...")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.pending_response = True

# Check if we need to generate a response (from suggestion button or chat input)
needs_response = st.session_state.pending_response and st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user"

if needs_response:
    st.session_state.pending_response = False

    # Build context
    context_parts = []
    if st.session_state.get("last_score"):
        s = st.session_state.last_score
        context_parts.append(
            f"Recent suitability: {s.get('energy_type', 'solar')} at "
            f"({s.get('latitude', 0):.2f}, {s.get('longitude', 0):.2f}), "
            f"factor_score={s.get('factor_score', 0):.1%}, "
            f"ml_score={s.get('ml_score', 'N/A')}"
        )
    if st.session_state.get("last_climate_risk"):
        r = st.session_state.last_climate_risk
        context_parts.append(f"Recent climate risk: score={r['risk_score']:.3f}")

    api_messages = []
    system_msg = """You are the AI assistant for O3 EartH — a geospatial site suitability assessment system for renewable energy infrastructure.

About O3 EartH:
- O3 stands for the three pillars of sustainability: Environmental, Social, Economic
- Built as thesis research at Northeastern University

System Architecture:
- Factor Engine: Rule-based scoring using real-time data from NASA POWER (solar irradiance, wind speed, temperature, cloud cover), Open-Elevation (terrain slope), Open-Meteo Flood (river discharge), and USGS Earthquake (seismic activity)
- OlmoEarth ML: Allen Institute's geospatial foundation model (97M params) extracts 768-dim embeddings from Sentinel-2 satellite imagery. XGBoost classifier trained on 8,000 globally distributed samples scores how much a location resembles existing energy infrastructure sites. Spatial cross-validation AUC: 0.867.
- Climate Risk: NASA POWER current data + IPCC AR6 SSP scenario projections (SSP126 through SSP585)
- LLM Chat: NVIDIA NIM for natural language interaction
- EIA Data: US Energy Information Administration API for plant data

MCP (Model Context Protocol) Tools:
- score_suitability: Score any lat/lon for solar or wind suitability (returns factor engine score + ML score)
- assess_climate_risk: Climate risk assessment with SSP scenario projections
- query_eia: Query US energy plant database
- analyze: LLM-powered analysis of results
- generate_report: Create formatted reports from analysis data

Key research finding: OlmoEarth embeddings improve suitability prediction from AUC 0.579 (geography only) to 0.902 (with embeddings). Under spatial cross-validation (leave-one-country-out), AUC is 0.867 — demonstrating genuine generalization.

Be concise and specific. When users ask about the system, explain using the architecture above."""

    if context_parts:
        system_msg += "\n\nUser's recent analyses:\n" + "\n".join(context_parts)
    api_messages.append({"role": "system", "content": system_msg})

    recent = st.session_state.chat_messages[-10:]
    api_messages.extend({"role": m["role"], "content": m["content"]} for m in recent)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api.llm_chat(
                api_messages,
                max_tokens=4096,
                temperature=0.7,
            )

        if result:
            response = result["text"]
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        else:
            st.error("Failed to get a response.")

# Clear chat
if st.session_state.chat_messages:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
