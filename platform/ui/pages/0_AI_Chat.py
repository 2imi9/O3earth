"""AI Chat — interactive chat with LLM."""

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

# ------------------------------------------------------------------
# Check LLM status
# ------------------------------------------------------------------
llm_info = api.llm_status()

PROVIDER_LABELS = {
    "nvidia_nim": "NVIDIA NIM (Cloud)",
    "vllm": "Local GPU (vLLM)",
}

available_providers: list[dict] = []
if llm_info and llm_info.get("providers"):
    for p in llm_info["providers"]:
        label = PROVIDER_LABELS.get(p["name"], p["name"])
        if p["available"]:
            label += f"  --  {p['model']}"
        else:
            label += "  --  unavailable"
        available_providers.append({
            "name": p["name"],
            "label": label,
            "available": p["available"],
            "supports_tools": p.get("supports_tools", False),
            "model": p.get("model", "none"),
        })

if llm_info and llm_info.get("available"):
    active = [p["label"] for p in available_providers if p["available"]]
    st.success(f"LLM available: {', '.join(active)}")
else:
    st.warning("No LLM providers available. Add `NVIDIA_API_KEY` to `.env`")

st.divider()

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
with st.expander("Settings"):
    provider_options = ["auto"] + [p["name"] for p in available_providers]
    provider_display = {
        "auto": "Auto (best available)",
        **{p["name"]: p["label"] for p in available_providers},
    }
    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        format_func=lambda x: provider_display.get(x, x),
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 8192, 8192, 256)
    enable_tools = False  # Disabled for now

# ------------------------------------------------------------------
# Quick actions
# ------------------------------------------------------------------
quick_col1, quick_col2 = st.columns(2)

with quick_col1:
    if st.button("Analyze Last Suitability", use_container_width=True):
        if st.session_state.get("last_suitability"):
            s = st.session_state.last_suitability
            prompt = f"Analyze this suitability result: {s.get('energy_type', 'solar')} at ({s.get('latitude', 0):.2f}, {s.get('longitude', 0):.2f}), score={s.get('combined_score', 0):.1%}, factors: {s.get('factors', {})}"
            st.session_state.setdefault("chat_messages", [])
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.rerun()
        else:
            st.info("Run a suitability score first.")

with quick_col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about site suitability, energy data, climate risk..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context
    context_parts = []
    if st.session_state.get("last_suitability"):
        s = st.session_state.last_suitability
        context_parts.append(
            f"Recent suitability: {s.get('energy_type', 'solar')} at ({s.get('latitude', 0):.2f}, {s.get('longitude', 0):.2f}), "
            f"score={s.get('combined_score', 0):.1%}"
        )
    if st.session_state.get("last_climate_risk"):
        r = st.session_state.last_climate_risk
        context_parts.append(f"Recent climate risk: score={r['risk_score']:.3f}")

    api_messages = []
    system_msg = "You are an AI assistant for O3 EartH, a geospatial site suitability assessment system for renewable energy."
    if context_parts:
        system_msg += "\n\nUser's recent analyses:\n" + "\n".join(context_parts)
    api_messages.append({"role": "system", "content": system_msg})

    recent = st.session_state.chat_messages[-10:]
    api_messages.extend({"role": m["role"], "content": m["content"]} for m in recent)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api.llm_chat(
                api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_tools=enable_tools,
                provider=selected_provider,
            )

        if result:
            response = result["text"]
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.caption(f"{result.get('provider', '?')} | {result.get('model', '?')}")
        else:
            st.error("Failed to get a response.")
