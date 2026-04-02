"""
Cloud LLM Client using NVIDIA NIM (OpenAI-compatible API).

Drop-in replacement for VLLMClient — same interface (chat, analyze_asset,
analyze_climate_risk, generate_report, query).

Author: Zim (Millennium Fellowship Research)
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai not installed. pip install openai")

from .client import ChatMessage, GenerationResult, AgenticResult


@dataclass
class CloudConfig:
    """Configuration for NVIDIA NIM cloud LLM."""
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "openai/gpt-oss-20b"
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    system_prompt: str = (
        "You are an AI assistant specialized in renewable energy analysis, "
        "climate risk assessment, and asset valuation. You have expertise in "
        "satellite imagery analysis, EIA data, NEMS-based projections, "
        "NPV, IRR, and LCOE calculations. "
        "Provide clear, data-driven insights."
    )

    @classmethod
    def from_env(cls) -> "CloudConfig":
        return cls(
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
            base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            model=os.environ.get("NVIDIA_MODEL", "openai/gpt-oss-20b"),
            max_tokens=int(os.environ.get("NVIDIA_MAX_TOKENS", "8192")),
            temperature=float(os.environ.get("NVIDIA_TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("NVIDIA_TOP_P", "0.95")),
        )


class CloudLLMClient:
    """Cloud LLM client using NVIDIA NIM (OpenAI-compatible API).

    Matches the VLLMClient interface so it's a drop-in replacement.
    """

    def __init__(self, config: Optional[CloudConfig] = None):
        if not HAS_OPENAI:
            raise ImportError("openai package required. pip install openai")

        self.config = config or CloudConfig.from_env()
        if not self.config.api_key:
            raise ValueError("NVIDIA_API_KEY not set")

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self.provider = "nvidia_nim"
        self.model = self.config.model
        logger.info(f"NVIDIA NIM client initialized: {self.model}")

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> GenerationResult:
        """Generate text completion."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def chat(self, messages: List[ChatMessage], max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> GenerationResult:
        """Chat completion with message history."""
        api_messages = []

        # Add system prompt if not already present
        has_system = any(m.role == "system" for m in messages)
        if not has_system:
            api_messages.append({"role": "system", "content": self.config.system_prompt})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Ensure first non-system message is from user
        non_system = [m for m in api_messages if m["role"] != "system"]
        if not non_system or non_system[0]["role"] != "user":
            insert_idx = next(
                (i for i, m in enumerate(api_messages) if m["role"] != "system"),
                len(api_messages),
            )
            api_messages.insert(insert_idx, {"role": "user", "content": "Hello."})

        response = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            messages=api_messages,
        )
        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def analyze_asset(self, asset_data: Dict[str, Any], include_recommendations: bool = True) -> str:
        """Analyze a renewable energy asset."""
        prompt = f"""Analyze the following renewable energy asset:

Asset Data:
{_format_dict(asset_data)}

Provide a comprehensive analysis including:
1. Asset overview and key characteristics
2. Financial metrics interpretation (NPV, IRR, LCOE)
3. Risk factors and considerations
4. Climate impact assessment
{"5. Investment recommendations" if include_recommendations else ""}

Be specific and data-driven in your analysis."""
        result = self.chat([ChatMessage("user", prompt)])
        return result.text

    def analyze_climate_risk(self, risk_data: Dict[str, Any], location: Optional[str] = None) -> str:
        """Analyze climate risk assessment results."""
        location_str = f" for {location}" if location else ""
        prompt = f"""Analyze the following climate risk assessment{location_str}:

Risk Data:
{_format_dict(risk_data)}

Provide analysis of:
1. Overall risk interpretation
2. Key risk factors and their implications
3. Resource availability assessment (solar GHI, wind speed)
4. Climate change projections impact
5. Mitigation recommendations"""
        result = self.chat([ChatMessage("user", prompt)])
        return result.text

    def generate_report(self, data: Dict[str, Any], report_type: str = "valuation", format: str = "markdown") -> str:
        """Generate a formatted report."""
        report_instructions = {
            "valuation": "Create a professional asset valuation report with executive summary, financial analysis, risk assessment, and conclusions.",
            "climate": "Create a climate risk report with location overview, risk factors, resource assessment, and adaptation recommendations.",
            "detection": "Create a detection report summarizing satellite imagery analysis results.",
            "summary": "Create a brief executive summary highlighting key findings and recommendations.",
        }
        instruction = report_instructions.get(report_type, report_instructions["summary"])
        prompt = f"""{instruction}

Data:
{_format_dict(data)}

Format the report in {'Markdown' if format == 'markdown' else 'plain text'}."""
        result = self.chat([ChatMessage("user", prompt)], max_tokens=4096)
        return result.text

    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Answer a natural language question."""
        if context:
            prompt = f"""Context:
{_format_dict(context)}

Question: {question}

Provide a clear, concise answer based on the context."""
        else:
            prompt = question
        result = self.chat([ChatMessage("user", prompt)])
        return result.text

    def agentic_chat(
        self,
        messages: List[ChatMessage],
        tool_handlers,
        tools: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_rounds: int = 5,
    ) -> "AgenticResult":
        """Chat with tool calling loop.

        Sends messages + tools to NIM. If the model calls tools,
        executes them via tool_handlers and feeds results back.
        Falls back to plain chat if the model doesn't support tools.
        """
        api_messages = self._build_messages(messages, agentic=True)
        tool_calls_log = []

        for _ in range(max_rounds):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    messages=api_messages,
                    tools=tools,
                )
            except Exception as e:
                logger.warning(f"Tool calling failed, falling back: {e}")
                return self._plain_fallback(api_messages, max_tokens, temperature, tool_calls_log)

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                usage = response.usage
                return AgenticResult(
                    text=msg.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    finish_reason=choice.finish_reason or "stop",
                    tool_calls=tool_calls_log,
                )

            # Append assistant message with tool calls
            api_messages.append(msg.model_dump())

            # Execute each tool call
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info(f"Tool call: {tc.function.name}({args})")
                result = tool_handlers.execute_tool(tc.function.name, args)
                tool_calls_log.append({"tool": tc.function.name, "arguments": args, "result": result})

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        return AgenticResult(
            text="[Max tool rounds reached]",
            prompt_tokens=0, completion_tokens=0,
            finish_reason="max_rounds", tool_calls=tool_calls_log,
        )

    def _build_messages(self, messages: List[ChatMessage], agentic: bool = False) -> list:
        api_messages = []
        has_system = any(m.role == "system" for m in messages)
        if not has_system:
            prompt = self.config.system_prompt
            if agentic:
                prompt += (
                    "\n\nYou have tools for renewable energy detection, climate risk "
                    "assessment, asset valuation, and EIA data queries. Use them when "
                    "the user's question needs real data or calculations."
                )
            api_messages.append({"role": "system", "content": prompt})
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})
        non_system = [m for m in api_messages if m["role"] != "system"]
        if not non_system or non_system[0]["role"] != "user":
            idx = next((i for i, m in enumerate(api_messages) if m["role"] != "system"), len(api_messages))
            api_messages.insert(idx, {"role": "user", "content": "Hello."})
        return api_messages

    def _plain_fallback(self, api_messages, max_tokens, temperature, tool_calls_log):
        response = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            messages=api_messages,
        )
        choice = response.choices[0]
        usage = response.usage
        return AgenticResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls_log,
        )

    def is_loaded(self) -> bool:
        return True

    def unload(self):
        pass


def _format_dict(d: Dict[str, Any]) -> str:
    """Format dictionary for prompt inclusion."""
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        elif isinstance(value, list):
            lines.append(f"{key}: {value[:5]}{'...' if len(value) > 5 else ''}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def create_cloud_client(config: Optional[CloudConfig] = None) -> Optional[CloudLLMClient]:
    """Factory to create NVIDIA NIM client. Returns None if not configured."""
    if not HAS_OPENAI:
        return None
    try:
        return CloudLLMClient(config)
    except (ValueError, Exception) as e:
        logger.warning(f"NVIDIA NIM client unavailable: {e}")
        return None
