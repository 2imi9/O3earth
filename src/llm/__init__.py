"""
LLM Integration Module for OpenEnergy Engine

Provides LLM inference via NVIDIA NIM (cloud) or local vLLM for:
- Natural language queries about assets and climate data
- Automated analysis and report generation
- Agent workflow orchestration

Usage:
    from src.llm.cloud_client import create_cloud_client

    # Create NVIDIA NIM client (reads NVIDIA_API_KEY from env)
    client = create_cloud_client()

    # Generate text
    result = client.generate("Explain solar capacity factors")

    # Analyze asset
    analysis = client.analyze_asset(asset_data)

Author: Zim (Millennium Fellowship Research)
"""

from .config import VLLMConfig
from .client import (
    VLLMClient,
    MockVLLMClient,
    ChatMessage,
    GenerationResult,
    create_vllm_client,
    HAS_VLLM,
)
from .cloud_client import (
    CloudConfig,
    CloudLLMClient,
    create_cloud_client,
)
from .prompts import (
    PromptType,
    PromptTemplate,
    SYSTEM_PROMPTS,
    ASSET_ANALYSIS_PROMPT,
    CLIMATE_RISK_PROMPT,
    DETECTION_PROMPT,
    REPORT_TEMPLATES,
    get_system_prompt,
    format_extreme_events,
    build_analysis_prompt,
)

__all__ = [
    # Config
    "VLLMConfig",
    "CloudConfig",
    # Clients
    "VLLMClient",
    "MockVLLMClient",
    "CloudLLMClient",
    "ChatMessage",
    "GenerationResult",
    "create_vllm_client",
    "create_cloud_client",
    "HAS_VLLM",
    # Prompts
    "PromptType",
    "PromptTemplate",
    "SYSTEM_PROMPTS",
    "ASSET_ANALYSIS_PROMPT",
    "CLIMATE_RISK_PROMPT",
    "DETECTION_PROMPT",
    "REPORT_TEMPLATES",
    "get_system_prompt",
    "format_extreme_events",
    "build_analysis_prompt",
]
