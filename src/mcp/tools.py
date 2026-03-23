"""
MCP Tool Definitions for OpenEnergy Engine

Defines tools for:
- Renewable energy detection from satellite imagery
- Climate risk assessment
- Asset valuation
- EIA data queries
- LLM-powered analysis

Author: Zim (Millennium Fellowship Research)
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Input/Output Schemas
# =============================================================================

@dataclass
class DetectionInput:
    """Input for renewable energy detection."""
    latitude: float
    longitude: float
    date_range: str = "2024-01-01/2024-12-31"
    max_cloud_cover: float = 10.0


@dataclass
class DetectionOutput:
    """Output from renewable energy detection."""
    detected: bool
    detection_confidence: float
    classification: str
    classification_confidence: float
    estimated_capacity_mw: float
    image_date: str
    image_source: str


@dataclass
class ClimateRiskInput:
    """Input for climate risk assessment."""
    latitude: float
    longitude: float
    elevation: float = 0.0
    asset_type: str = "solar"
    scenario: str = "SSP245"
    target_year: int = 2050


@dataclass
class ClimateRiskOutput:
    """Output from climate risk assessment."""
    risk_score: float
    solar_ghi_p50: float
    wind_speed_p50: float
    extreme_event_probs: Dict[str, float]
    temperature_change_c: float
    precipitation_change_pct: float


@dataclass
class ValuationInput:
    """Input for asset valuation."""
    asset_id: str
    asset_type: str
    latitude: float
    longitude: float
    state: str
    capacity_mw: float
    capacity_factor: float = 0.25
    verification_status: str = "pending"
    verification_confidence: float = 0.0
    discount_rate: float = 0.08


@dataclass
class ValuationOutput:
    """Output from asset valuation."""
    npv_usd: float
    irr: float
    lcoe_per_mwh: float
    payback_years: float
    risk_adjusted_npv: float
    annual_generation_mwh: List[float]
    annual_revenue_usd: List[float]


@dataclass
class EIAQueryInput:
    """Input for EIA data query."""
    query_type: str  # "generators", "generation", "prices", "capacity"
    state: Optional[str] = None
    energy_source: Optional[str] = None  # "SUN", "WND", etc.
    min_capacity_mw: float = 1.0
    scenario: str = "ref2025"


@dataclass
class AnalysisInput:
    """Input for LLM analysis."""
    question: str
    context: Optional[Dict[str, Any]] = None
    analysis_type: str = "general"  # "asset", "climate", "comparison"


# =============================================================================
# Tool Handlers
# =============================================================================

class ToolHandlers:
    """
    Handlers for MCP tools.

    Each handler wraps the corresponding module functionality
    and returns structured output.
    """

    def __init__(self, config=None):
        """Initialize handlers with optional config."""
        self.config = config
        self._model = None
        self._climate_model = None
        self._valuation_engine = None
        self._eia_client = None
        self._satellite_client = None
        self._llm_client = None

    # -------------------------------------------------------------------------
    # Detection Tools
    # -------------------------------------------------------------------------

    def detect_renewable(self, input_data: DetectionInput) -> DetectionOutput:
        """Detect renewable energy installations from satellite imagery.

        Args:
            input_data: Detection input parameters

        Returns:
            DetectionOutput with detection results
        """
        # Lazy load model
        if self._model is None:
            try:
                from src.models import create_model
                self._model = create_model(task="multi")
                logger.info("Detection model loaded")
            except Exception as e:
                logger.error(f"Failed to load detection model: {e}")
                raise

        # Get satellite imagery
        if self._satellite_client is None:
            try:
                from src.data import create_satellite_client
                self._satellite_client = create_satellite_client()
            except ImportError:
                from src.data.satellite import MockSatelliteClient
                self._satellite_client = MockSatelliteClient()
                logger.warning("Using mock satellite client")

        # Fetch imagery
        chip = self._satellite_client.get_sentinel2_chip(
            lat=input_data.latitude,
            lon=input_data.longitude,
            date_range=input_data.date_range,
            max_cloud_cover=input_data.max_cloud_cover,
        )

        if chip is None:
            return DetectionOutput(
                detected=False,
                detection_confidence=0.0,
                classification="none",
                classification_confidence=0.0,
                estimated_capacity_mw=0.0,
                image_date="N/A",
                image_source="N/A",
            )

        # Run detection
        import torch
        image_tensor = torch.from_numpy(chip.normalize()).unsqueeze(0).float()

        # Pad to 12 channels if needed
        if image_tensor.shape[1] < 12:
            padding = torch.zeros(1, 12 - image_tensor.shape[1], *image_tensor.shape[2:])
            image_tensor = torch.cat([image_tensor, padding], dim=1)

        with torch.no_grad():
            outputs = self._model(image_tensor)

        # Parse outputs
        detection_prob = outputs["detection"].item()
        classification_logits = outputs["classification"][0]
        class_idx = classification_logits.argmax().item()
        class_names = ["none", "solar_pv", "solar_thermal", "wind_onshore", "wind_offshore", "hydro"]
        classification = class_names[class_idx] if class_idx < len(class_names) else "unknown"

        return DetectionOutput(
            detected=detection_prob > 0.5,
            detection_confidence=detection_prob,
            classification=classification,
            classification_confidence=torch.softmax(classification_logits, dim=0)[class_idx].item(),
            estimated_capacity_mw=outputs["capacity_mw"].item(),
            image_date=chip.datetime,
            image_source=chip.source,
        )

    # -------------------------------------------------------------------------
    # Climate Tools
    # -------------------------------------------------------------------------

    def assess_climate_risk(self, input_data: ClimateRiskInput) -> ClimateRiskOutput:
        """Assess climate risk for a location.

        Args:
            input_data: Climate risk input parameters

        Returns:
            ClimateRiskOutput with risk assessment
        """
        # Lazy load model
        if self._climate_model is None:
            from src.models import create_climate_model, ClimateScenario
            self._climate_model = create_climate_model()
            logger.info("Climate model loaded")

        from src.models import ClimateScenario

        # Map scenario string to enum
        scenario_map = {
            "SSP126": ClimateScenario.SSP126,
            "SSP245": ClimateScenario.SSP245,
            "SSP370": ClimateScenario.SSP370,
            "SSP585": ClimateScenario.SSP585,
        }
        scenario = scenario_map.get(input_data.scenario.upper(), ClimateScenario.SSP245)

        # Run assessment
        risk = self._climate_model.assess_risk(
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            elevation=input_data.elevation,
            asset_type=input_data.asset_type,
            scenario=scenario,
            target_year=input_data.target_year,
        )

        return ClimateRiskOutput(
            risk_score=risk.risk_score,
            solar_ghi_p50=risk.solar_ghi_kwh_m2_year["p50"],
            wind_speed_p50=risk.wind_speed_m_s["p50"],
            extreme_event_probs=risk.extreme_event_probs,
            temperature_change_c=risk.temperature_change_c,
            precipitation_change_pct=risk.precipitation_change_pct,
        )

    # -------------------------------------------------------------------------
    # Valuation Tools
    # -------------------------------------------------------------------------

    def value_asset(self, input_data: ValuationInput) -> ValuationOutput:
        """Calculate asset valuation.

        Args:
            input_data: Valuation input parameters

        Returns:
            ValuationOutput with valuation results
        """
        # Lazy load engine
        if self._valuation_engine is None:
            from src.valuation import ValuationEngine
            self._valuation_engine = ValuationEngine(discount_rate=input_data.discount_rate)
            logger.info("Valuation engine loaded")

        from src.valuation import AssetCharacteristics, AssetType

        # Map asset type string to enum
        type_map = {
            "solar_utility": AssetType.SOLAR_UTILITY,
            "solar_distributed": AssetType.SOLAR_DISTRIBUTED,
            "wind_onshore": AssetType.WIND_ONSHORE,
            "wind_offshore": AssetType.WIND_OFFSHORE,
            "hydro": AssetType.HYDRO,
        }
        asset_type = type_map.get(input_data.asset_type.lower(), AssetType.SOLAR_UTILITY)

        # Create asset
        asset = AssetCharacteristics(
            asset_id=input_data.asset_id,
            asset_type=asset_type,
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            state=input_data.state,
            capacity_mw=input_data.capacity_mw,
            capacity_factor=input_data.capacity_factor,
            verification_status=input_data.verification_status,
            verification_confidence=input_data.verification_confidence,
        )

        # Run valuation
        result = self._valuation_engine.value_asset(asset)

        return ValuationOutput(
            npv_usd=result.npv_usd,
            irr=result.irr,
            lcoe_per_mwh=result.lcoe_per_mwh,
            payback_years=result.payback_years,
            risk_adjusted_npv=result.risk_adjusted_npv,
            annual_generation_mwh=result.annual_generation_mwh[:5],  # First 5 years
            annual_revenue_usd=result.annual_revenue_usd[:5],
        )

    # -------------------------------------------------------------------------
    # EIA Tools
    # -------------------------------------------------------------------------

    def query_eia(self, input_data: EIAQueryInput) -> Dict[str, Any]:
        """Query EIA database.

        Args:
            input_data: EIA query parameters

        Returns:
            Query results as dictionary
        """
        # Lazy load client
        if self._eia_client is None:
            try:
                from src.eia import EIAClient
                self._eia_client = EIAClient()
                logger.info("EIA client initialized")
            except ValueError as e:
                logger.warning(f"EIA client not available: {e}")
                return {"error": str(e), "hint": "Set EIA_API_KEY environment variable"}

        query_type = input_data.query_type.lower()

        if query_type == "generators":
            df = self._eia_client.get_operating_generators(
                state=input_data.state,
                energy_source=input_data.energy_source,
                min_capacity_mw=input_data.min_capacity_mw,
            )
            return {
                "count": len(df),
                "data": df.head(20).to_dict(orient="records") if not df.empty else [],
            }

        elif query_type == "generation":
            if input_data.state:
                df = self._eia_client.get_state_generation(
                    state=input_data.state,
                    fuel_type=input_data.energy_source,
                )
                return {
                    "count": len(df),
                    "data": df.head(20).to_dict(orient="records") if not df.empty else [],
                }

        elif query_type == "prices":
            df = self._eia_client.get_electricity_price_forecast(
                scenario=input_data.scenario,
            )
            return {
                "count": len(df),
                "data": df.to_dict(orient="records") if not df.empty else [],
            }

        elif query_type == "capacity":
            source = "solar" if input_data.energy_source == "SUN" else "wind"
            df = self._eia_client.get_renewable_capacity_forecast(
                energy_source=source,
                scenario=input_data.scenario,
            )
            return {
                "count": len(df),
                "data": df.to_dict(orient="records") if not df.empty else [],
            }

        elif query_type == "summary":
            if input_data.state:
                return self._eia_client.get_state_renewable_summary(input_data.state)

        return {"error": f"Unknown query type: {query_type}"}

    # -------------------------------------------------------------------------
    # LLM Tools
    # -------------------------------------------------------------------------

    def analyze(self, input_data: AnalysisInput) -> str:
        """Run LLM analysis.

        Args:
            input_data: Analysis input parameters

        Returns:
            Analysis text from LLM
        """
        # Lazy load LLM client (NVIDIA NIM preferred, vLLM fallback)
        if self._llm_client is None:
            from src.llm.cloud_client import create_cloud_client
            self._llm_client = create_cloud_client()
            if self._llm_client is None:
                from src.llm import create_vllm_client
                self._llm_client = create_vllm_client()
            logger.info("LLM client initialized")

        if input_data.analysis_type == "asset" and input_data.context:
            return self._llm_client.analyze_asset(input_data.context)
        elif input_data.analysis_type == "climate" and input_data.context:
            return self._llm_client.analyze_climate_risk(input_data.context)
        else:
            return self._llm_client.query(input_data.question, input_data.context)

    def generate_report(
        self,
        data: Dict[str, Any],
        report_type: str = "summary",
        format: str = "markdown"
    ) -> str:
        """Generate a formatted report.

        Args:
            data: Data to include in report
            report_type: Type of report
            format: Output format

        Returns:
            Formatted report
        """
        if self._llm_client is None:
            from src.llm.cloud_client import create_cloud_client
            self._llm_client = create_cloud_client()
            if self._llm_client is None:
                from src.llm import create_vllm_client
                self._llm_client = create_vllm_client()

        return self._llm_client.generate_report(data, report_type, format)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name. Returns result as dict."""
        dispatch = {
            "detect_renewable": (self.detect_renewable, DetectionInput),
            "assess_climate_risk": (self.assess_climate_risk, ClimateRiskInput),
            "value_asset": (self.value_asset, ValuationInput),
            "query_eia": (self.query_eia, EIAQueryInput),
        }
        if tool_name not in dispatch:
            return {"error": f"Unknown tool: {tool_name}"}
        handler_fn, input_cls = dispatch[tool_name]
        try:
            input_data = input_cls(**arguments)
            result = handler_fn(input_data)
            if hasattr(result, "__dataclass_fields__"):
                return asdict(result)
            elif isinstance(result, dict):
                return result
            return {"result": str(result)}
        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return {"error": str(e)}


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_DEFINITIONS = {
    "detect_renewable": {
        "name": "detect_renewable",
        "description": "Detect renewable energy installations (solar, wind) from satellite imagery at a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "date_range": {"type": "string", "description": "Date range for imagery (YYYY-MM-DD/YYYY-MM-DD)"},
                "max_cloud_cover": {"type": "number", "description": "Maximum cloud cover percentage"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    "assess_climate_risk": {
        "name": "assess_climate_risk",
        "description": "Assess climate risk and resource availability for renewable energy at a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "elevation": {"type": "number", "description": "Elevation in meters"},
                "asset_type": {"type": "string", "enum": ["solar", "wind"], "description": "Type of renewable energy"},
                "scenario": {"type": "string", "enum": ["SSP126", "SSP245", "SSP370", "SSP585"], "description": "Climate scenario"},
                "target_year": {"type": "integer", "description": "Target year for projections"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    "value_asset": {
        "name": "value_asset",
        "description": "Calculate NPV, IRR, and other valuation metrics for a renewable energy asset",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "Unique asset identifier"},
                "asset_type": {"type": "string", "enum": ["solar_utility", "solar_distributed", "wind_onshore", "wind_offshore", "hydro"]},
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "state": {"type": "string", "description": "US state code (e.g., CA, TX)"},
                "capacity_mw": {"type": "number", "description": "Installed capacity in MW"},
                "capacity_factor": {"type": "number", "description": "Expected capacity factor (0-1)"},
                "verification_status": {"type": "string", "enum": ["verified", "pending", "flagged"]},
                "verification_confidence": {"type": "number", "description": "Verification confidence (0-1)"},
            },
            "required": ["asset_id", "asset_type", "latitude", "longitude", "state", "capacity_mw"],
        },
    },
    "query_eia": {
        "name": "query_eia",
        "description": "Query EIA database for generator data, generation statistics, price forecasts, or capacity projections",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "enum": ["generators", "generation", "prices", "capacity", "summary"]},
                "state": {"type": "string", "description": "US state code"},
                "energy_source": {"type": "string", "enum": ["SUN", "WND", "WAT", "GEO"], "description": "Energy source code"},
                "min_capacity_mw": {"type": "number", "description": "Minimum capacity filter"},
                "scenario": {"type": "string", "description": "AEO scenario (e.g., ref2025)"},
            },
            "required": ["query_type"],
        },
    },
    "analyze": {
        "name": "analyze",
        "description": "Use LLM to analyze data, answer questions, or provide insights about renewable energy assets",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question or analysis request"},
                "context": {"type": "object", "description": "Optional context data"},
                "analysis_type": {"type": "string", "enum": ["general", "asset", "climate", "comparison"]},
            },
            "required": ["question"],
        },
    },
    "generate_report": {
        "name": "generate_report",
        "description": "Generate a formatted report from analysis data",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to include in report"},
                "report_type": {"type": "string", "enum": ["summary", "valuation", "climate", "detection"]},
                "format": {"type": "string", "enum": ["markdown", "text", "json"]},
            },
            "required": ["data"],
        },
    },
}


def get_tool_definitions(enabled_categories: List[str] = None) -> Dict[str, Dict]:
    """Get tool definitions filtered by enabled categories.

    Args:
        enabled_categories: List of enabled categories
            ("detection", "climate", "valuation", "eia", "llm")

    Returns:
        Filtered tool definitions
    """
    if enabled_categories is None:
        return TOOL_DEFINITIONS

    category_tools = {
        "detection": ["detect_renewable"],
        "climate": ["assess_climate_risk"],
        "valuation": ["value_asset"],
        "eia": ["query_eia"],
        "llm": ["analyze", "generate_report"],
    }

    enabled_tools = set()
    for category in enabled_categories:
        enabled_tools.update(category_tools.get(category, []))

    return {k: v for k, v in TOOL_DEFINITIONS.items() if k in enabled_tools}


def get_openai_tools() -> List[Dict]:
    """Convert TOOL_DEFINITIONS to OpenAI function calling format.

    Excludes 'analyze' and 'generate_report' to avoid LLM recursion.
    """
    skip = {"analyze", "generate_report"}
    result = []
    for name, defn in TOOL_DEFINITIONS.items():
        if name in skip:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": defn["name"],
                "description": defn["description"],
                "parameters": defn["input_schema"],
            },
        })
    return result
