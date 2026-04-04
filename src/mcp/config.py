"""
MCP Server Configuration for OpenEnergy Engine

Configuration for the Model Context Protocol server that exposes
renewable energy analysis tools.

Author: Zim (Millennium Fellowship Research)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MCPConfig:
    """Configuration for MCP server.

    Usage:
        config = MCPConfig()
        config = MCPConfig.from_env()
        config = MCPConfig(port=8080, enable_llm=False)
    """

    # Server settings
    name: str = "openenergy-engine"
    version: str = "0.1.0"
    port: int = 3000
    host: str = "localhost"

    # Tool enablement
    enable_detection_tools: bool = True
    enable_climate_tools: bool = True
    enable_valuation_tools: bool = True
    enable_eia_tools: bool = True
    enable_satellite_tools: bool = True

    # LLM integration
    enable_llm: bool = True
    llm_model: str = "google/gemma-4-E2B-it"

    # Caching
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = 24

    # Logging
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Timeout settings (seconds)
    model_timeout: int = 120
    eia_timeout: int = 30
    satellite_timeout: int = 60

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables.

        Environment variables:
            MCP_NAME: Server name
            MCP_PORT: Server port (default: 3000)
            MCP_HOST: Server host (default: localhost)
            MCP_ENABLE_LLM: Enable LLM features (default: true)
            MCP_LLM_MODEL: LLM model ID
            MCP_LOG_LEVEL: Logging level
            MCP_CACHE_DIR: Cache directory path
        """
        def parse_bool(val: str) -> bool:
            return val.lower() in ("true", "1", "yes")

        return cls(
            name=os.environ.get("MCP_NAME", "openenergy-engine"),
            port=int(os.environ.get("MCP_PORT", "3000")),
            host=os.environ.get("MCP_HOST", "localhost"),
            enable_detection_tools=parse_bool(os.environ.get("MCP_ENABLE_DETECTION", "true")),
            enable_climate_tools=parse_bool(os.environ.get("MCP_ENABLE_CLIMATE", "true")),
            enable_valuation_tools=parse_bool(os.environ.get("MCP_ENABLE_VALUATION", "true")),
            enable_eia_tools=parse_bool(os.environ.get("MCP_ENABLE_EIA", "true")),
            enable_satellite_tools=parse_bool(os.environ.get("MCP_ENABLE_SATELLITE", "true")),
            enable_llm=parse_bool(os.environ.get("MCP_ENABLE_LLM", "true")),
            llm_model=os.environ.get("MCP_LLM_MODEL", "google/gemma-4-E2B-it"),
            log_level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
            cache_dir=os.environ.get("MCP_CACHE_DIR"),
        )

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool categories."""
        tools = []
        if self.enable_detection_tools:
            tools.append("detection")
        if self.enable_climate_tools:
            tools.append("climate")
        if self.enable_valuation_tools:
            tools.append("valuation")
        if self.enable_eia_tools:
            tools.append("eia")
        if self.enable_satellite_tools:
            tools.append("satellite")
        if self.enable_llm:
            tools.append("llm")
        return tools
