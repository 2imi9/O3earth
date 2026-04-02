"""
MCP Server Module for OpenEnergy Engine

Model Context Protocol server exposing renewable energy analysis tools:
- Satellite detection of solar/wind installations
- Climate risk assessment
- Asset valuation with NEMS projections
- EIA data queries
- LLM-powered analysis

Usage:
    from src.mcp import create_mcp_server, MCPConfig

    # Create and run with defaults
    server = create_mcp_server()
    server.run()

    # Or with custom config
    config = MCPConfig(port=8080, enable_llm=False)
    server = create_mcp_server(config)
    server.run()

CLI Usage:
    python -m src.mcp.server
    python -m src.mcp.server --port 8080 --no-llm

Author: Zim (Millennium Fellowship Research)
"""

from .config import MCPConfig
from .tools import (
    ToolHandlers,
    DetectionInput,
    DetectionOutput,
    ClimateRiskInput,
    ClimateRiskOutput,
    ValuationInput,
    ValuationOutput,
    EIAQueryInput,
    AnalysisInput,
    TOOL_DEFINITIONS,
    get_tool_definitions,
)

# Conditional import for server (requires fastmcp)
try:
    from .server import MCPServer, create_mcp_server, HAS_FASTMCP
except ImportError:
    HAS_FASTMCP = False
    MCPServer = None
    create_mcp_server = None

__all__ = [
    # Config
    "MCPConfig",
    # Server
    "MCPServer",
    "create_mcp_server",
    "HAS_FASTMCP",
    # Tools
    "ToolHandlers",
    "DetectionInput",
    "DetectionOutput",
    "ClimateRiskInput",
    "ClimateRiskOutput",
    "ValuationInput",
    "ValuationOutput",
    "EIAQueryInput",
    "AnalysisInput",
    "TOOL_DEFINITIONS",
    "get_tool_definitions",
]
