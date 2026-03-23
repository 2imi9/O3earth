"""
vLLM Configuration for OpenEnergy Engine

Provides configuration for local LLM inference using vLLM.

Author: Zim (Millennium Fellowship Research)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class VLLMConfig:
    """Configuration for vLLM inference.

    Supports embedded local inference with Qwen3-8B as default.

    Usage:
        config = VLLMConfig()
        config = VLLMConfig.from_env()
        config = VLLMConfig(model_id="meta-llama/Llama-3.1-8B-Instruct")
    """

    # Model settings
    model_id: str = "Qwen/Qwen3-8B"
    dtype: str = "float16"  # float16, bfloat16, or auto
    trust_remote_code: bool = True

    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Hardware settings
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None  # Auto-detect if None

    # Sampling settings
    seed: Optional[int] = None

    # Chat template
    chat_template: Optional[str] = None  # Use model default if None

    # System prompt for analysis tasks
    system_prompt: str = field(default_factory=lambda: (
        "You are an AI assistant specialized in renewable energy analysis, "
        "climate risk assessment, and asset valuation. You have access to "
        "satellite imagery analysis, EIA data, and NEMS-based projections. "
        "Provide clear, data-driven insights."
    ))

    @classmethod
    def from_env(cls) -> "VLLMConfig":
        """Load configuration from environment variables.

        Environment variables:
            VLLM_MODEL: Model ID (default: Qwen/Qwen3-8B)
            VLLM_DTYPE: Data type (default: float16)
            VLLM_GPU_MEMORY: GPU memory utilization (default: 0.9)
            VLLM_MAX_TOKENS: Max generation tokens (default: 2048)
            VLLM_TEMPERATURE: Sampling temperature (default: 0.7)
            VLLM_TENSOR_PARALLEL: Tensor parallel size (default: 1)
        """
        return cls(
            model_id=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
            dtype=os.environ.get("VLLM_DTYPE", "float16"),
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY", "0.9")),
            max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("VLLM_TEMPERATURE", "0.7")),
            tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL", "1")),
        )

    def to_vllm_kwargs(self) -> dict:
        """Convert to kwargs for vLLM LLM constructor."""
        kwargs = {
            "model": self.model_id,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        return kwargs

    def to_sampling_params(self) -> dict:
        """Convert to kwargs for vLLM SamplingParams."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.seed is not None:
            params["seed"] = self.seed
        return params
