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

if not st.session_state.chat_messages:
    st.markdown("**Try asking:**")
    suggestions = [
        "What makes a good solar farm location?",
        "Compare solar potential in Sahara vs Northern Europe",
        "What climate risks affect wind farms?",
        "Explain how OlmoEarth embeddings work for site scoring",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, use_container_width=True, key=f"suggest_{i}"):
            st.session_state.chat_messages.append({"role": "user", "content": suggestion})
            st.rerun()
    st.divider()

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about site suitability, energy data, climate risk..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context from recent analyses
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
    system_msg = (
        "You are an AI assistant for O3 EartH, a geospatial site suitability "
        "assessment system for renewable energy infrastructure. You help users "
        "understand suitability scores, climate risks, and energy data. "
        "Be concise and specific."
    )
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

# Clear chat button at bottom
if st.session_state.chat_messages:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
