"""FastAPI application for O3 EartH."""

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to sys.path so `src.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Add platform/ to sys.path so `api.*` imports work
_platform_root = str(Path(__file__).resolve().parent.parent)
if _platform_root not in sys.path:
    sys.path.insert(0, _platform_root)

from dotenv import load_dotenv

# Load .env file BEFORE anything reads os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.mcp.tools import ToolHandlers
from src.mcp.config import MCPConfig
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# Global singleton — persists models across requests
handlers: ToolHandlers | None = None


def _warmup_vllm():
    """Pre-load vLLM model in a background thread so it's ready for requests."""
    try:
        from src.llm.client import HAS_VLLM
        if not HAS_VLLM:
            return
        from api.routes.llm import _init_vllm
        client = _init_vllm()
        if client:
            logger.info("Warming up vLLM model (loading into GPU)...")
            client._ensure_loaded()
            logger.info(f"vLLM model loaded: {client.model}")
    except Exception as e:
        logger.warning(f"vLLM warmup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handlers
    config = MCPConfig.from_env()
    handlers = ToolHandlers(config)
    logger.info("ToolHandlers initialized")

    # Warm up vLLM in background so it doesn't block the first request
    warmup = threading.Thread(target=_warmup_vllm, daemon=True)
    warmup.start()

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="OpenEnergy Engine API",
    version="0.1.0",
    description="Renewable energy detection, climate risk, and valuation API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include route modules
from api.routes import detection, climate, valuation, eia, llm  # noqa: E402

app.include_router(detection.router, prefix="/api", tags=["detection"])
app.include_router(climate.router, prefix="/api", tags=["climate"])
app.include_router(valuation.router, prefix="/api", tags=["valuation"])
app.include_router(eia.router, prefix="/api", tags=["eia"])
app.include_router(llm.router, prefix="/api", tags=["llm"])


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Check API health and module availability."""
    modules = {
        "detection": True,
        "climate": True,
        "valuation": True,
        "eia": False,
        "satellite": False,
        "llm": False,
    }

    # Check EIA availability
    try:
        import os
        modules["eia"] = bool(os.environ.get("EIA_API_KEY"))
    except Exception:
        pass

    # Check satellite deps
    try:
        import planetary_computer  # noqa: F401
        modules["satellite"] = True
    except ImportError:
        pass

    # Check LLM availability (NVIDIA NIM or local vLLM)
    try:
        import os as _os
        if _os.environ.get("NVIDIA_API_KEY"):
            modules["llm"] = True
        else:
            try:
                from src.llm.client import HAS_VLLM
                if HAS_VLLM:
                    modules["llm"] = True
            except Exception:
                pass
    except Exception:
        pass

    return HealthResponse(status="ok", modules=modules)
