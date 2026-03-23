"""AI Chat — interactive chat with selectable LLM provider."""

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
st.caption("Renewable energy analysis assistant")

api = get_api_client()

# ------------------------------------------------------------------
# Check LLM status & build provider list
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

# Status banner
if llm_info and llm_info.get("available"):
    active = [p["label"] for p in available_providers if p["available"]]
    st.success(f"LLM available: {', '.join(active)}")
else:
    st.warning(
        "No LLM providers available.\n\n"
        "- **Cloud**: Add `NVIDIA_API_KEY` to `.env`\n"
        "- **Local**: Install vLLM with a CUDA GPU"
    )

st.divider()

# ------------------------------------------------------------------
# Chat settings
# ------------------------------------------------------------------
with st.expander("Settings"):
    # Provider selector
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

    # Show warning if selected provider is unavailable
    if selected_provider != "auto":
        info = next((p for p in available_providers if p["name"] == selected_provider), None)
        if info and not info["available"]:
            st.error(f"'{PROVIDER_LABELS.get(selected_provider, selected_provider)}' is not available.")
        elif info and not info["supports_tools"]:
            st.info("Tool calling is not supported with this provider and will be auto-disabled.")

    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 8192, 8192, 256)

    # Auto-disable tool toggle when provider doesn't support it
    selected_info = next((p for p in available_providers if p["name"] == selected_provider), None)
    tools_supported = selected_provider == "auto" or (selected_info and selected_info.get("supports_tools", False))
    enable_tools = st.toggle(
        "Enable tool calling",
        value=tools_supported,
        disabled=not tools_supported,
    )

# ------------------------------------------------------------------
# Quick analysis buttons
# ------------------------------------------------------------------
st.subheader("Quick Analysis")
quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    if st.button("Analyze Last Valuation", use_container_width=True):
        if st.session_state.last_valuation:
            result = api.llm_analyze(
                analysis_type="asset",
                context=st.session_state.last_valuation,
                provider=selected_provider,
            )
            if result:
                st.session_state.setdefault("chat_messages", [])
                st.session_state.chat_messages.append({"role": "user", "content": "Analyze my last valuation result."})
                st.session_state.chat_messages.append({"role": "assistant", "content": result["text"]})
        else:
            st.info("Run a valuation first on the Asset Valuation page.")

with quick_col2:
    if st.button("Analyze Last Climate Risk", use_container_width=True):
        if st.session_state.last_climate_risk:
            result = api.llm_analyze(
                analysis_type="climate",
                context=st.session_state.last_climate_risk,
                provider=selected_provider,
            )
            if result:
                st.session_state.setdefault("chat_messages", [])
                st.session_state.chat_messages.append({"role": "user", "content": "Analyze my last climate risk assessment."})
                st.session_state.chat_messages.append({"role": "assistant", "content": result["text"]})
        else:
            st.info("Run a climate risk assessment first.")

with quick_col3:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------------
st.subheader("Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        # Show tool calls if present
        if msg.get("tool_calls"):
            with st.expander(f"Tools used ({len(msg['tool_calls'])})", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(f"**{tc['tool']}**")
                    st.json(tc.get("arguments", {}))
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about renewable energy, climate risk, valuations..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context from recent analyses
    context_parts = []
    if st.session_state.last_valuation:
        v = st.session_state.last_valuation
        context_parts.append(
            f"Recent valuation: NPV=${v['npv_usd']:,.0f}, IRR={v['irr']*100:.1f}%, "
            f"LCOE=${v['lcoe_per_mwh']:.2f}/MWh, asset={v['asset_id']}"
        )
    if st.session_state.last_climate_risk:
        r = st.session_state.last_climate_risk
        context_parts.append(
            f"Recent climate risk: score={r['risk_score']:.3f}, "
            f"solar_ghi_p50={r['solar_ghi_kwh_m2_year']['p50']:.0f}"
        )

    api_messages = []
    if context_parts:
        system_msg = (
            "You are an AI assistant specialized in renewable energy analysis. "
            "Here is context from the user's recent analyses:\n" + "\n".join(context_parts)
        )
        api_messages.append({"role": "system", "content": system_msg})

    # Only send role + content to the API (strip tool_calls, etc.)
    recent = st.session_state.chat_messages[-10:]
    api_messages.extend({"role": m["role"], "content": m["content"]} for m in recent)

    # Get response
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
            tool_calls = result.get("tool_calls", [])

            # Show tool calls
            if tool_calls:
                with st.expander(f"Tools used ({len(tool_calls)})", expanded=True):
                    for tc in tool_calls:
                        st.markdown(f"**{tc['tool']}**")
                        st.json(tc.get("arguments", {}))
                        if "error" in tc.get("result", {}):
                            st.error(tc["result"]["error"])

            st.markdown(response)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls,
            })
            st.caption(f"Provider: {result.get('provider', '?')} | Model: {result.get('model', '?')} | Tokens: {result.get('completion_tokens', '?')}")
        else:
            st.error("Failed to get a response. Is the LLM running?")
