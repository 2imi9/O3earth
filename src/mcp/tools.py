"""
MCP Tool Definitions for O3 EartH

Tools:
- Site suitability scoring (factor engine + ML model)
- Climate risk assessment
- EIA data queries
- LLM-powered analysis
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Input/Output Schemas
# =============================================================================

@dataclass
class SuitabilityInput:
    """Input for site suitability scoring."""
    latitude: float
    longitude: float
    energy_type: str = "solar"  # solar, wind, hydro, geothermal


@dataclass
class SuitabilityOutput:
    """Output from site suitability scoring."""
    factor_score: float
    ml_score: Optional[float]
    combined_score: float
    energy_type: str
    confidence: str
    factors: Dict[str, float]
    ml_available: bool


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
class EIAQueryInput:
    """Input for EIA data query."""
    query_type: str  # "generators", "generation", "prices", "capacity"
    state: Optional[str] = None
    energy_source: Optional[str] = None
    min_capacity_mw: float = 1.0
    scenario: str = "ref2025"


@dataclass
class AnalysisInput:
    """Input for LLM analysis."""
    question: str
    context: Optional[Dict[str, Any]] = None
    analysis_type: str = "general"


# =============================================================================
# Tool Handlers
# =============================================================================

class ToolHandlers:
    """Handlers for MCP tools."""

    def __init__(self, config=None):
        self.config = config
        self._climate_model = None
        self._eia_client = None
        self._llm_client = None
        self._suitability_engine = None
        self._xgb_models = {}
        self._xgb_scalers = {}

    # -------------------------------------------------------------------------
    # Suitability Tools
    # -------------------------------------------------------------------------

    def score_suitability(self, input_data: SuitabilityInput) -> SuitabilityOutput:
        """Score a location for renewable energy suitability.

        Uses both the configurable factor engine and trained ML model.
        """
        from src.scoring.engine import SuitabilityEngine

        engine = SuitabilityEngine(input_data.energy_type)
        result = engine.score(input_data.latitude, input_data.longitude)
        factor_score = result.overall_score

        # Try ML model
        ml_score = None
        ml_available = False
        try:
            ml_score = self._get_ml_score(
                input_data.latitude, input_data.longitude, input_data.energy_type
            )
            if ml_score is not None:
                ml_available = True
        except Exception as e:
            logger.debug(f"ML score unavailable: {e}")

        combined = 0.6 * ml_score + 0.4 * factor_score if ml_score is not None else factor_score

        if combined > 0.7:
            confidence = "high"
        elif combined > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return SuitabilityOutput(
            factor_score=factor_score,
            ml_score=ml_score,
            combined_score=combined,
            energy_type=input_data.energy_type,
            confidence=confidence,
            factors=result.factor_scores,
            ml_available=ml_available,
        )

    def _get_ml_score(self, lat: float, lon: float, energy_type: str) -> Optional[float]:
        """Get ML model score using nearest pre-computed embedding."""
        import numpy as np
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / f"results/suitability/xgb_{energy_type}.json"
        scaler_path = project_root / f"results/suitability/scaler_{energy_type}.pkl"
        meta_path = project_root / "data/embeddings_shuffled/embeddings_meta.csv"
        emb_path = project_root / "data/embeddings_shuffled/embeddings.npy"

        if not all(p.exists() for p in [model_path, scaler_path, meta_path, emb_path]):
            return None

        import pandas as pd
        meta = pd.read_csv(str(meta_path))
        emb = np.load(str(emb_path))

        dist = np.sqrt((meta["lat"] - lat)**2 + (meta["lon"] - lon)**2)
        nearest_idx = dist.idxmin()
        if dist[nearest_idx] > 0.5:
            return None

        if energy_type not in self._xgb_models:
            import xgboost as xgb
            import pickle
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            with open(str(scaler_path), "rb") as f:
                scaler = pickle.load(f)
            self._xgb_models[energy_type] = model
            self._xgb_scalers[energy_type] = scaler

        embedding = emb[nearest_idx].reshape(1, -1)
        embedding_s = self._xgb_scalers[energy_type].transform(embedding)
        return float(self._xgb_models[energy_type].predict_proba(embedding_s)[0, 1])

    # -------------------------------------------------------------------------
    # Climate Tools
    # -------------------------------------------------------------------------

    def assess_climate_risk(self, input_data: ClimateRiskInput) -> ClimateRiskOutput:
        """Assess climate risk for a location."""
        if self._climate_model is None:
            from src.models import create_climate_model
            self._climate_model = create_climate_model()

        from src.models import ClimateScenario
        scenario_map = {
            "SSP126": ClimateScenario.SSP126,
            "SSP245": ClimateScenario.SSP245,
            "SSP370": ClimateScenario.SSP370,
            "SSP585": ClimateScenario.SSP585,
        }
        scenario = scenario_map.get(input_data.scenario.upper(), ClimateScenario.SSP245)

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
    # EIA Tools
    # -------------------------------------------------------------------------

    def query_eia(self, input_data: EIAQueryInput) -> Dict[str, Any]:
        """Query EIA database."""
        if self._eia_client is None:
            try:
                from src.eia import EIAClient
                self._eia_client = EIAClient()
            except (ValueError, ImportError) as e:
                return {"error": str(e), "hint": "Set EIA_API_KEY environment variable"}

        query_type = input_data.query_type.lower()

        if query_type == "generators":
            df = self._eia_client.get_operating_generators(
                state=input_data.state,
                energy_source=input_data.energy_source,
                min_capacity_mw=input_data.min_capacity_mw,
            )
            return {"count": len(df), "data": df.head(20).to_dict(orient="records") if not df.empty else []}
        elif query_type == "prices":
            df = self._eia_client.get_electricity_price_forecast(scenario=input_data.scenario)
            return {"count": len(df), "data": df.to_dict(orient="records") if not df.empty else []}
        elif query_type == "summary" and input_data.state:
            return self._eia_client.get_state_renewable_summary(input_data.state)

        return {"error": f"Unknown query type: {query_type}"}

    # -------------------------------------------------------------------------
    # LLM Tools
    # -------------------------------------------------------------------------

    def analyze(self, input_data: AnalysisInput) -> str:
        """Run LLM analysis."""
        if self._llm_client is None:
            from src.llm.cloud_client import create_cloud_client
            self._llm_client = create_cloud_client()
            if self._llm_client is None:
                from src.llm import create_vllm_client
                self._llm_client = create_vllm_client()

        return self._llm_client.query(input_data.question, input_data.context)

    def generate_report(self, data: Dict[str, Any], report_type: str = "summary", format: str = "markdown") -> str:
        """Generate a formatted report."""
        if self._llm_client is None:
            from src.llm.cloud_client import create_cloud_client
            self._llm_client = create_cloud_client()
            if self._llm_client is None:
                from src.llm import create_vllm_client
                self._llm_client = create_vllm_client()
        return self._llm_client.generate_report(data, report_type, format)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        dispatch = {
            "score_suitability": (self.score_suitability, SuitabilityInput),
            "assess_climate_risk": (self.assess_climate_risk, ClimateRiskInput),
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
    "score_suitability": {
        "name": "score_suitability",
        "description": "Score a location for renewable energy suitability using OlmoEarth embeddings and configurable factor engine",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "energy_type": {"type": "string", "enum": ["solar", "wind", "hydro", "geothermal"], "description": "Type of renewable energy"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    "assess_climate_risk": {
        "name": "assess_climate_risk",
        "description": "Assess climate risk and resource availability for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "scenario": {"type": "string", "enum": ["SSP126", "SSP245", "SSP370", "SSP585"]},
                "target_year": {"type": "integer", "description": "Target year for projections"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    "query_eia": {
        "name": "query_eia",
        "description": "Query EIA database for generator data, price forecasts, or capacity projections",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "enum": ["generators", "prices", "summary"]},
                "state": {"type": "string", "description": "US state code"},
                "energy_source": {"type": "string", "enum": ["SUN", "WND", "WAT", "GEO"]},
            },
            "required": ["query_type"],
        },
    },
    "analyze": {
        "name": "analyze",
        "description": "Use LLM to analyze data or answer questions about site suitability",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question or analysis request"},
                "context": {"type": "object", "description": "Optional context data"},
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
                "report_type": {"type": "string", "enum": ["summary", "suitability", "climate"]},
                "format": {"type": "string", "enum": ["markdown", "text", "json"]},
            },
            "required": ["data"],
        },
    },
}


def get_tool_definitions(enabled_categories: List[str] = None) -> Dict[str, Dict]:
    """Get tool definitions filtered by enabled categories."""
    if enabled_categories is None:
        return TOOL_DEFINITIONS

    category_tools = {
        "suitability": ["score_suitability"],
        "climate": ["assess_climate_risk"],
        "eia": ["query_eia"],
        "llm": ["analyze", "generate_report"],
    }

    enabled_tools = set()
    for category in enabled_categories:
        enabled_tools.update(category_tools.get(category, []))

    return {k: v for k, v in TOOL_DEFINITIONS.items() if k in enabled_tools}


def get_openai_tools() -> List[Dict]:
    """Convert TOOL_DEFINITIONS to OpenAI function calling format."""
    skip = {"analyze", "generate_report"}
    return [
        {"type": "function", "function": {"name": d["name"], "description": d["description"], "parameters": d["input_schema"]}}
        for name, d in TOOL_DEFINITIONS.items() if name not in skip
    ]
