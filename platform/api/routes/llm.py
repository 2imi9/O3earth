"""LLM chat and analysis routes — supports NVIDIA NIM (cloud) and local vLLM."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 8192
    temperature: float = 1.0
    enable_tools: bool = True
    provider: str = Field("auto", pattern="^(auto|nvidia_nim|vllm)$")


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    tool_calls: List[Dict[str, Any]] = []


class AnalysisRequest(BaseModel):
    analysis_type: str = Field("general", pattern="^(general|asset|climate|report)$")
    question: str = ""
    context: Optional[Dict[str, Any]] = None
    report_type: str = "summary"
    provider: str = Field("auto", pattern="^(auto|nvidia_nim|vllm)$")


class AnalysisResponse(BaseModel):
    text: str
    analysis_type: str


class ProviderStatus(BaseModel):
    name: str
    available: bool
    model: str = "none"
    supports_tools: bool = False


class LLMStatusResponse(BaseModel):
    available: bool
    provider: str
    model: str
    providers: List[ProviderStatus] = []


# ---------------------------------------------------------------------------
# Provider registry (lazy-loaded singletons per provider)
# ---------------------------------------------------------------------------

_clients: Dict[str, Any] = {}
_checked: Dict[str, bool] = {"nvidia_nim": False, "vllm": False}
_tool_handlers = None


def _init_nvidia_nim():
    """Try to initialise NVIDIA NIM cloud client. Returns client or None."""
    if _checked.get("nvidia_nim"):
        return _clients.get("nvidia_nim")
    _checked["nvidia_nim"] = True
    try:
        from src.llm.cloud_client import create_cloud_client
        client = create_cloud_client()
        if client:
            _clients["nvidia_nim"] = client
            logger.info(f"NVIDIA NIM ready: {client.model}")
            return client
    except Exception as e:
        logger.warning(f"NVIDIA NIM not available: {e}")
    return None


def _init_vllm():
    """Try to initialise local vLLM client. Returns client or None."""
    if _checked.get("vllm"):
        return _clients.get("vllm")
    _checked["vllm"] = True
    try:
        from src.llm.client import create_vllm_client, HAS_VLLM, MockVLLMClient
        if not HAS_VLLM:
            return None
        client = create_vllm_client()
        # Don't expose the mock — only real vLLM
        if isinstance(client, MockVLLMClient):
            return None
        _clients["vllm"] = client
        logger.info(f"vLLM ready: {client.model}")
        return client
    except Exception as e:
        logger.warning(f"vLLM not available: {e}")
    return None


def _get_client(provider: str = "auto"):
    """Get LLM client for the requested provider.

    provider values:
      - "nvidia_nim": Force NVIDIA NIM cloud
      - "vllm":       Force local vLLM
      - "auto":       Try NIM first, then vLLM (backward-compatible default)
    """
    if provider == "nvidia_nim":
        return _init_nvidia_nim()
    elif provider == "vllm":
        return _init_vllm()
    else:  # auto
        client = _init_nvidia_nim()
        if client:
            return client
        return _init_vllm()


def _get_tool_handlers():
    global _tool_handlers
    if _tool_handlers is not None:
        return _tool_handlers
    try:
        from api.main import handlers
        if handlers:
            _tool_handlers = handlers
            return _tool_handlers
    except Exception:
        pass
    from src.mcp.tools import ToolHandlers
    _tool_handlers = ToolHandlers()
    return _tool_handlers


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/llm/status", response_model=LLMStatusResponse)
async def llm_status():
    """Return availability of all LLM providers."""
    providers = []

    nim_client = _init_nvidia_nim()
    providers.append(ProviderStatus(
        name="nvidia_nim",
        available=nim_client is not None,
        model=getattr(nim_client, "model", "none") if nim_client else "none",
        supports_tools=True,
    ))

    vllm_client = _init_vllm()
    providers.append(ProviderStatus(
        name="vllm",
        available=vllm_client is not None,
        model=getattr(vllm_client, "model", "none") if vllm_client else "none",
        supports_tools=True,
    ))

    # Backward-compatible defaults: prefer NIM
    default_client = nim_client or vllm_client
    return LLMStatusResponse(
        available=default_client is not None,
        provider=getattr(default_client, "provider", "none") if default_client else "none",
        model=getattr(default_client, "model", "none") if default_client else "none",
        providers=providers,
    )


@router.post("/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    client = _get_client(req.provider)
    if client is None:
        hints = {
            "nvidia_nim": "Set NVIDIA_API_KEY in .env.",
            "vllm": "Install vllm and ensure an NVIDIA GPU is available.",
            "auto": "Set NVIDIA_API_KEY in .env or install vllm with a GPU.",
        }
        raise HTTPException(503, f"LLM provider '{req.provider}' not available. {hints.get(req.provider, '')}")

    from src.llm.client import ChatMessage
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

    actual_provider = getattr(client, "provider", "unknown")
    # Only enable tools if the client supports agentic_chat
    enable_tools = req.enable_tools and hasattr(client, "agentic_chat")

    try:
        if enable_tools:
            from src.mcp.tools import get_openai_tools
            tools = get_openai_tools()
            handlers = _get_tool_handlers()
            result = client.agentic_chat(
                messages, handlers, tools,
                max_tokens=req.max_tokens, temperature=req.temperature,
            )
            return ChatResponse(
                text=result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                model=getattr(client, "model", "unknown"),
                provider=actual_provider,
                tool_calls=result.tool_calls,
            )

        # Plain chat (works for both CloudLLMClient and VLLMClient)
        result = client.chat(messages, max_tokens=req.max_tokens, temperature=req.temperature)
        return ChatResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            model=getattr(client, "model", "unknown"),
            provider=actual_provider,
        )
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")


@router.post("/llm/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    client = _get_client(req.provider)
    if client is None:
        hints = {
            "nvidia_nim": "Set NVIDIA_API_KEY in .env.",
            "vllm": "Install vllm and ensure an NVIDIA GPU is available.",
            "auto": "Set NVIDIA_API_KEY in .env or install vllm with a GPU.",
        }
        raise HTTPException(503, f"LLM provider '{req.provider}' not available. {hints.get(req.provider, '')}")
    try:
        if req.analysis_type == "asset" and req.context:
            text = client.analyze_asset(req.context)
        elif req.analysis_type == "climate" and req.context:
            text = client.analyze_climate_risk(req.context)
        elif req.analysis_type == "report" and req.context:
            text = client.generate_report(req.context, req.report_type)
        else:
            text = client.query(req.question, req.context)
        return AnalysisResponse(text=text, analysis_type=req.analysis_type)
    except Exception as e:
        raise HTTPException(500, f"Analysis error: {str(e)}")
