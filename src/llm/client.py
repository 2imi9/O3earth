"""
vLLM Client for OpenEnergy Engine

Provides local LLM inference using vLLM for:
- Natural language queries about assets and climate data
- Automated analysis and report generation
- Agent workflow orchestration

Author: Zim (Millennium Fellowship Research)
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

from .config import VLLMConfig

logger = logging.getLogger(__name__)

# Check for vLLM availability
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    logger.warning(
        "vLLM not installed. Install with: pip install vllm\n"
        "LLM features will be disabled. MCP tools will still work."
    )


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class GenerationResult:
    """Result of LLM generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass
class AgenticResult:
    """Result from agentic chat with tool calling."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class VLLMClient:
    """
    Client for local LLM inference using vLLM.

    Features:
    - Lazy model loading (loads on first use)
    - Chat and completion APIs
    - Specialized methods for asset analysis and report generation

    Usage:
        client = VLLMClient()

        # Simple generation
        result = client.generate("Explain solar capacity factors")

        # Chat
        messages = [
            ChatMessage("user", "What's the NPV of a 100MW solar farm?")
        ]
        result = client.chat(messages)

        # Asset analysis
        analysis = client.analyze_asset(asset_data)
    """

    def __init__(self, config: Optional[VLLMConfig] = None):
        """Initialize vLLM client.

        Args:
            config: VLLMConfig instance. Uses defaults if None.
        """
        if not HAS_VLLM:
            raise ImportError(
                "vLLM is required for VLLMClient. Install with: pip install vllm"
            )

        self.config = config or VLLMConfig()
        self._llm: Optional[LLM] = None
        self._loaded = False
        self.provider = "vllm"
        self.model = self.config.model_id

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        logger.info(f"Loading vLLM model: {self.config.model_id}")
        self._llm = LLM(**self.config.to_vllm_kwargs())
        self._loaded = True
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional sampling params

        Returns:
            GenerationResult with generated text
        """
        self._ensure_loaded()

        # Build sampling params
        params = self.config.to_sampling_params()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params.update(kwargs)

        sampling_params = SamplingParams(**params)

        # Generate
        outputs = self._llm.generate([prompt], sampling_params)
        output = outputs[0]

        return GenerationResult(
            text=output.outputs[0].text,
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        """Chat completion with message history.

        Args:
            messages: List of ChatMessage objects
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            GenerationResult with assistant response
        """
        self._ensure_loaded()

        # Format messages for chat
        formatted_messages = []

        # Add system prompt if not present
        if not any(m.role == "system" for m in messages):
            formatted_messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Use tokenizer to apply chat template
        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def analyze_asset(
        self,
        asset_data: Dict[str, Any],
        include_recommendations: bool = True
    ) -> str:
        """Analyze a renewable energy asset.

        Args:
            asset_data: Asset characteristics and valuation data
            include_recommendations: Include investment recommendations

        Returns:
            Analysis text
        """
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

        messages = [ChatMessage("user", prompt)]
        result = self.chat(messages)
        return result.text

    def analyze_climate_risk(
        self,
        risk_data: Dict[str, Any],
        location: Optional[str] = None
    ) -> str:
        """Analyze climate risk assessment results.

        Args:
            risk_data: Climate risk output data
            location: Optional location description

        Returns:
            Risk analysis text
        """
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

        messages = [ChatMessage("user", prompt)]
        result = self.chat(messages)
        return result.text

    def generate_report(
        self,
        data: Dict[str, Any],
        report_type: str = "valuation",
        format: str = "markdown"
    ) -> str:
        """Generate a formatted report.

        Args:
            data: Data to include in report
            report_type: "valuation", "climate", "detection", or "summary"
            format: "markdown", "text", or "json"

        Returns:
            Formatted report
        """
        report_instructions = {
            "valuation": "Create a professional asset valuation report with sections for executive summary, financial analysis, risk assessment, and conclusions.",
            "climate": "Create a climate risk report with sections for location overview, risk factors, resource assessment, and adaptation recommendations.",
            "detection": "Create a detection report summarizing satellite imagery analysis results, verified installations, and confidence levels.",
            "summary": "Create a brief executive summary highlighting key findings and recommendations.",
        }

        instruction = report_instructions.get(report_type, report_instructions["summary"])
        format_instruction = {
            "markdown": "Format the report in Markdown with headers, bullet points, and tables where appropriate.",
            "text": "Format as plain text with clear sections.",
            "json": "Return a structured JSON object with report sections.",
        }.get(format, "")

        prompt = f"""{instruction}

Data:
{_format_dict(data)}

{format_instruction}"""

        messages = [ChatMessage("user", prompt)]
        result = self.chat(messages, max_tokens=4096)
        return result.text

    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Answer a natural language question.

        Args:
            question: User question
            context: Optional context data

        Returns:
            Answer text
        """
        if context:
            prompt = f"""Context:
{_format_dict(context)}

Question: {question}

Provide a clear, concise answer based on the context."""
        else:
            prompt = question

        messages = [ChatMessage("user", prompt)]
        result = self.chat(messages)
        return result.text

    def agentic_chat(
        self,
        messages: List[ChatMessage],
        tool_handlers,
        tools: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_rounds: int = 5,
    ) -> AgenticResult:
        """Chat with tool calling loop (Gemma 4-style function calling).

        The model generates ``<tool_call>...</tool_call>`` blocks which are
        parsed, executed via *tool_handlers*, and fed back as tool-role
        messages until the model produces a final text answer.
        """
        self._ensure_loaded()
        tokenizer = self._llm.get_tokenizer()

        # Build initial message list
        formatted: List[Dict[str, Any]] = []
        if not any(m.role == "system" for m in messages):
            sys_content = (
                self.config.system_prompt
                + "\n\nYou have tools for renewable energy detection, climate risk "
                "assessment, asset valuation, and EIA data queries. Use them when "
                "the user's question needs real data or calculations."
            )
            formatted.append({"role": "system", "content": sys_content})

        for msg in messages:
            formatted.append({"role": msg.role, "content": msg.content})

        tool_calls_log: List[Dict[str, Any]] = []
        total_prompt = 0
        total_completion = 0

        for _ in range(max_rounds):
            # Apply chat template with tools
            try:
                prompt = tokenizer.apply_chat_template(
                    formatted,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback for models whose template ignores tools
                prompt = tokenizer.apply_chat_template(
                    formatted,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            result = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            total_prompt += result.prompt_tokens
            total_completion += result.completion_tokens
            text = result.text.strip()

            # Parse <tool_call>…</tool_call> blocks
            tc_matches = re.findall(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                text,
                re.DOTALL,
            )

            if not tc_matches:
                # No tool calls — clean up special tokens and return
                clean = re.sub(r"<\|.*?\|>", "", text).strip()
                return AgenticResult(
                    text=clean,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    finish_reason=result.finish_reason,
                    tool_calls=tool_calls_log,
                )

            # Append assistant turn (raw, including tool_call tags)
            formatted.append({"role": "assistant", "content": text})

            for tc_json in tc_matches:
                try:
                    tc_data = json.loads(tc_json)
                    tool_name = tc_data.get("name", "")
                    tool_args = tc_data.get("arguments", {})
                except json.JSONDecodeError:
                    continue

                logger.info(f"Tool call (vLLM): {tool_name}({tool_args})")
                tool_result = tool_handlers.execute_tool(tool_name, tool_args)
                tool_calls_log.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                })

                # Feed tool result back
                formatted.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, default=str),
                })

        return AgenticResult(
            text="[Max tool rounds reached]",
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            finish_reason="max_rounds",
            tool_calls=tool_calls_log,
        )

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload(self):
        """Unload model to free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded = False
            logger.info("Model unloaded")


class MockVLLMClient:
    """Mock client for testing without GPU/vLLM.

    Returns placeholder responses for all methods.
    """

    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig()
        self.provider = "vllm_mock"
        self.model = self.config.model_id
        logger.info("Using MockVLLMClient (vLLM not available)")

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        return GenerationResult(
            text="[Mock response - vLLM not available]",
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            finish_reason="mock",
        )

    def chat(self, messages: List[ChatMessage], **kwargs) -> GenerationResult:
        return self.generate(messages[-1].content if messages else "")

    def analyze_asset(self, asset_data: Dict[str, Any], **kwargs) -> str:
        return "[Mock analysis - vLLM not available. Install vLLM for LLM features.]"

    def analyze_climate_risk(self, risk_data: Dict[str, Any], **kwargs) -> str:
        return "[Mock climate analysis - vLLM not available.]"

    def generate_report(self, data: Dict[str, Any], **kwargs) -> str:
        return "[Mock report - vLLM not available.]"

    def query(self, question: str, **kwargs) -> str:
        return "[Mock response - vLLM not available.]"

    def is_loaded(self) -> bool:
        return False

    def unload(self):
        pass


def _format_dict(d: Dict[str, Any], indent: int = 2) -> str:
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


def create_vllm_client(
    config: Optional[VLLMConfig] = None,
    use_mock: bool = False
) -> Union[VLLMClient, MockVLLMClient]:
    """Factory function to create vLLM client.

    Args:
        config: VLLMConfig instance
        use_mock: Force mock client (for testing)

    Returns:
        VLLMClient or MockVLLMClient
    """
    if use_mock or not HAS_VLLM:
        return MockVLLMClient(config)
    return VLLMClient(config)


if __name__ == "__main__":
    print("Testing vLLM Client...")

    if HAS_VLLM:
        print("vLLM available - creating client")
        client = create_vllm_client()
        print(f"Client created: {type(client).__name__}")
        print(f"Model: {client.config.model_id}")
    else:
        print("vLLM not available - using mock client")
        client = create_vllm_client(use_mock=True)
        result = client.generate("Test prompt")
        print(f"Mock result: {result.text}")
