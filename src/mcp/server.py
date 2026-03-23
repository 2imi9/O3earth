"""
MCP Server for OpenEnergy Engine

Model Context Protocol server exposing renewable energy analysis tools:
- Satellite detection of solar/wind installations
- Climate risk assessment
- Asset valuation with NEMS projections
- EIA data queries
- LLM-powered analysis

Author: Zim (Millennium Fellowship Research)
"""

import logging
import json
from typing import Optional, Dict, Any
from dataclasses import asdict

from .config import MCPConfig
from .tools import (
    ToolHandlers,
    DetectionInput,
    ClimateRiskInput,
    ValuationInput,
    EIAQueryInput,
    AnalysisInput,
    get_tool_definitions,
    TOOL_DEFINITIONS,
)

logger = logging.getLogger(__name__)

# Check for FastMCP availability
try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    logger.warning(
        "FastMCP not installed. Install with: pip install fastmcp\n"
        "MCP server will not be available."
    )


class MCPServer:
    """
    MCP Server for OpenEnergy Engine.

    Exposes renewable energy analysis tools via the Model Context Protocol.

    Usage:
        # Create and run server
        server = MCPServer()
        server.run()

        # Or with custom config
        config = MCPConfig(port=8080, enable_llm=False)
        server = MCPServer(config)
        server.run()

        # Use as context manager
        with MCPServer() as server:
            server.run()
    """

    def __init__(self, config: Optional[MCPConfig] = None):
        """Initialize MCP server.

        Args:
            config: Server configuration. Uses defaults if None.
        """
        if not HAS_FASTMCP:
            raise ImportError(
                "FastMCP is required for MCPServer. Install with: pip install fastmcp"
            )

        self.config = config or MCPConfig()
        self.handlers = ToolHandlers(self.config)
        self._mcp: Optional[FastMCP] = None

        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))

    def _setup_mcp(self):
        """Initialize FastMCP instance with tools."""
        self._mcp = FastMCP(self.config.name)

        # Get enabled tools
        enabled_categories = self.config.get_enabled_tools()
        tools = get_tool_definitions(enabled_categories)

        logger.info(f"Registering tools: {list(tools.keys())}")

        # Register each tool
        if "detect_renewable" in tools:
            @self._mcp.tool()
            def detect_renewable(
                latitude: float,
                longitude: float,
                date_range: str = "2024-01-01/2024-12-31",
                max_cloud_cover: float = 10.0
            ) -> Dict[str, Any]:
                """Detect renewable energy installations from satellite imagery."""
                input_data = DetectionInput(
                    latitude=latitude,
                    longitude=longitude,
                    date_range=date_range,
                    max_cloud_cover=max_cloud_cover,
                )
                result = self.handlers.detect_renewable(input_data)
                return asdict(result)

        if "assess_climate_risk" in tools:
            @self._mcp.tool()
            def assess_climate_risk(
                latitude: float,
                longitude: float,
                elevation: float = 0.0,
                asset_type: str = "solar",
                scenario: str = "SSP245",
                target_year: int = 2050
            ) -> Dict[str, Any]:
                """Assess climate risk for renewable energy at a location."""
                input_data = ClimateRiskInput(
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    asset_type=asset_type,
                    scenario=scenario,
                    target_year=target_year,
                )
                result = self.handlers.assess_climate_risk(input_data)
                return asdict(result)

        if "value_asset" in tools:
            @self._mcp.tool()
            def value_asset(
                asset_id: str,
                asset_type: str,
                latitude: float,
                longitude: float,
                state: str,
                capacity_mw: float,
                capacity_factor: float = 0.25,
                verification_status: str = "pending",
                verification_confidence: float = 0.0,
                discount_rate: float = 0.08
            ) -> Dict[str, Any]:
                """Calculate NPV, IRR, and valuation metrics for a renewable energy asset."""
                input_data = ValuationInput(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    latitude=latitude,
                    longitude=longitude,
                    state=state,
                    capacity_mw=capacity_mw,
                    capacity_factor=capacity_factor,
                    verification_status=verification_status,
                    verification_confidence=verification_confidence,
                    discount_rate=discount_rate,
                )
                result = self.handlers.value_asset(input_data)
                return asdict(result)

        if "query_eia" in tools:
            @self._mcp.tool()
            def query_eia(
                query_type: str,
                state: str = None,
                energy_source: str = None,
                min_capacity_mw: float = 1.0,
                scenario: str = "ref2025"
            ) -> Dict[str, Any]:
                """Query EIA database for energy data."""
                input_data = EIAQueryInput(
                    query_type=query_type,
                    state=state,
                    energy_source=energy_source,
                    min_capacity_mw=min_capacity_mw,
                    scenario=scenario,
                )
                return self.handlers.query_eia(input_data)

        if "analyze" in tools:
            @self._mcp.tool()
            def analyze(
                question: str,
                context: Dict[str, Any] = None,
                analysis_type: str = "general"
            ) -> str:
                """Use LLM to analyze data or answer questions about renewable energy."""
                input_data = AnalysisInput(
                    question=question,
                    context=context,
                    analysis_type=analysis_type,
                )
                return self.handlers.analyze(input_data)

        if "generate_report" in tools:
            @self._mcp.tool()
            def generate_report(
                data: Dict[str, Any],
                report_type: str = "summary",
                format: str = "markdown"
            ) -> str:
                """Generate a formatted report from analysis data."""
                return self.handlers.generate_report(data, report_type, format)

        # Add prompts for common workflows
        @self._mcp.prompt()
        def analyze_location(latitude: float, longitude: float) -> str:
            """Comprehensive analysis workflow for a location."""
            return f"""Analyze the renewable energy potential at coordinates ({latitude}, {longitude}):

1. First, use detect_renewable to check for existing installations
2. Then, use assess_climate_risk to evaluate the location's climate profile
3. If an installation is detected, use value_asset to estimate its value
4. Finally, use analyze to synthesize the findings into actionable insights

Provide a comprehensive summary of your findings."""

        @self._mcp.prompt()
        def compare_assets(asset_ids: str) -> str:
            """Compare multiple renewable energy assets."""
            return f"""Compare the following renewable energy assets: {asset_ids}

For each asset:
1. Use query_eia to get plant details
2. Use assess_climate_risk to evaluate location risk
3. Use value_asset to calculate valuation metrics

Then use analyze to provide a comparative analysis and investment recommendation."""

    def run(self):
        """Start the MCP server."""
        if self._mcp is None:
            self._setup_mcp()

        logger.info(f"Starting MCP server: {self.config.name}")
        logger.info(f"Enabled tools: {self.config.get_enabled_tools()}")

        # Run the server
        self._mcp.run()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


def create_mcp_server(config: Optional[MCPConfig] = None) -> MCPServer:
    """Factory function to create MCP server.

    Args:
        config: Server configuration

    Returns:
        MCPServer instance
    """
    return MCPServer(config)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenEnergy Engine MCP Server")
    parser.add_argument("--port", type=int, default=3000, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM features")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    config = MCPConfig(
        port=args.port,
        host=args.host,
        enable_llm=not args.no_llm,
        log_level=args.log_level,
    )

    server = create_mcp_server(config)
    server.run()


if __name__ == "__main__":
    main()
