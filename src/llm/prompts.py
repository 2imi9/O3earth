"""
Prompt Templates for OpenEnergy Engine LLM Integration

Structured prompts for renewable energy analysis, climate risk,
valuation, and agent workflows.

Author: Zim (Millennium Fellowship Research)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class PromptType(Enum):
    """Types of analysis prompts."""
    ASSET_ANALYSIS = "asset_analysis"
    CLIMATE_RISK = "climate_risk"
    VALUATION = "valuation"
    DETECTION = "detection"
    COMPARISON = "comparison"
    WORKFLOW = "workflow"


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    template: str
    required_fields: List[str]
    optional_fields: List[str] = None

    def format(self, **kwargs) -> str:
        """Format template with provided values."""
        # Check required fields
        missing = [f for f in self.required_fields if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return self.template.format(**kwargs)


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    "default": """You are an AI assistant specialized in renewable energy analysis, climate risk assessment, and asset valuation. You have expertise in:

- Solar and wind energy installations
- Satellite imagery analysis for energy infrastructure
- Climate projections and extreme event risk
- Financial valuation using NEMS/EIA data
- NPV, IRR, and LCOE calculations

Provide clear, data-driven insights. When discussing financial metrics, explain their significance. When discussing risks, quantify them where possible.""",

    "analyst": """You are a senior renewable energy analyst with expertise in:

- Due diligence for renewable energy investments
- Technical assessment of solar and wind assets
- Climate risk modeling and adaptation strategies
- Market analysis using EIA projections

Your responses should be professional, thorough, and actionable. Include specific recommendations when appropriate.""",

    "technical": """You are a technical specialist in Earth observation and renewable energy verification. Your expertise includes:

- Satellite imagery interpretation (Sentinel-2, Landsat)
- Remote sensing for infrastructure detection
- Geospatial analysis for site assessment
- Change detection for construction monitoring

Focus on technical accuracy and methodology in your responses.""",

    "workflow": """You are an AI agent orchestrating renewable energy analysis workflows. You can:

1. Fetch and analyze satellite imagery for any location
2. Query EIA databases for plant information
3. Run climate risk assessments
4. Calculate asset valuations
5. Generate comprehensive reports

Think step-by-step and explain your reasoning. When a task requires multiple steps, outline them clearly before proceeding.""",
}


# =============================================================================
# Analysis Prompts
# =============================================================================

ASSET_ANALYSIS_PROMPT = PromptTemplate(
    name="asset_analysis",
    template="""Analyze the following renewable energy asset:

## Asset Information
- **Type**: {asset_type}
- **Location**: {latitude}, {longitude} ({state})
- **Capacity**: {capacity_mw} MW
- **Capacity Factor**: {capacity_factor:.1%}
- **Verification Status**: {verification_status} (Confidence: {verification_confidence:.1%})

## Financial Metrics
- **NPV**: ${npv:,.0f}
- **IRR**: {irr:.1%}
- **LCOE**: ${lcoe:.2f}/MWh
- **Payback Period**: {payback_years:.1f} years
- **Risk-Adjusted NPV**: ${risk_adjusted_npv:,.0f}

## Risk Adjustments
- Climate Risk Discount: {climate_discount:.1%}
- Verification Discount: {verification_discount:.1%}

Provide a comprehensive analysis including:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Financial Assessment**: Interpret the metrics and their implications
3. **Risk Analysis**: Evaluate the risk factors and their impact
4. **Recommendations**: Actionable next steps for investors/operators""",
    required_fields=[
        "asset_type", "latitude", "longitude", "state", "capacity_mw",
        "capacity_factor", "verification_status", "verification_confidence",
        "npv", "irr", "lcoe", "payback_years", "risk_adjusted_npv",
        "climate_discount", "verification_discount"
    ]
)


CLIMATE_RISK_PROMPT = PromptTemplate(
    name="climate_risk",
    template="""Analyze the climate risk assessment for a {asset_type} installation:

## Location
- Coordinates: {latitude}, {longitude}
- Elevation: {elevation} meters

## Risk Assessment
- **Overall Risk Score**: {risk_score:.2f}/1.0
- **Climate Scenario**: {scenario}
- **Target Year**: {target_year}

## Resource Assessment
- Solar GHI (P50): {solar_ghi_p50:.0f} kWh/m²/year
- Wind Speed (P50): {wind_speed_p50:.1f} m/s

## Extreme Event Probabilities
{extreme_events}

## Climate Projections
- Temperature Change: {temp_change:+.1f}°C
- Precipitation Change: {precip_change:+.1f}%

Provide analysis of:
1. **Risk Interpretation**: What the overall score means for this asset
2. **Resource Adequacy**: Assessment of solar/wind resource quality
3. **Extreme Event Exposure**: Key risks and their likelihood
4. **Climate Change Impact**: Long-term implications for asset performance
5. **Mitigation Strategies**: Recommendations to reduce identified risks""",
    required_fields=[
        "asset_type", "latitude", "longitude", "elevation",
        "risk_score", "scenario", "target_year",
        "solar_ghi_p50", "wind_speed_p50",
        "extreme_events", "temp_change", "precip_change"
    ]
)


DETECTION_PROMPT = PromptTemplate(
    name="detection",
    template="""Analyze the satellite detection results:

## Detection Output
- **Installation Detected**: {detected} (Confidence: {detection_confidence:.1%})
- **Classification**: {classification}
- **Estimated Capacity**: {estimated_capacity_mw:.1f} MW

## Image Information
- Source: {image_source}
- Date: {image_date}
- Cloud Cover: {cloud_cover:.1f}%
- Location: {latitude}, {longitude}

## Segmentation Results
{segmentation_summary}

Provide interpretation of:
1. **Detection Confidence**: Reliability of the detection result
2. **Classification Assessment**: Likely energy source type and confidence
3. **Capacity Estimate**: How the estimate compares to typical installations
4. **Verification Needs**: Additional data or imagery needed for confirmation
5. **Next Steps**: Recommended follow-up actions""",
    required_fields=[
        "detected", "detection_confidence", "classification",
        "estimated_capacity_mw", "image_source", "image_date",
        "cloud_cover", "latitude", "longitude", "segmentation_summary"
    ]
)


COMPARISON_PROMPT = PromptTemplate(
    name="comparison",
    template="""Compare the following {num_assets} renewable energy assets:

{asset_summaries}

Provide a comparative analysis:
1. **Summary Table**: Key metrics side-by-side
2. **Financial Comparison**: Which offers the best returns?
3. **Risk Comparison**: Which has the lowest risk profile?
4. **Location Analysis**: Geographic advantages/disadvantages
5. **Recommendation**: Ranking and investment priority""",
    required_fields=["num_assets", "asset_summaries"]
)


# =============================================================================
# Workflow Prompts
# =============================================================================

WORKFLOW_PROMPT = PromptTemplate(
    name="workflow",
    template="""Execute the following analysis workflow:

**Task**: {task_description}

**Available Tools**:
1. `detect_renewable(lat, lon)` - Detect installations from satellite imagery
2. `assess_climate_risk(lat, lon, scenario)` - Assess climate risks
3. `value_asset(asset_data)` - Calculate asset valuation
4. `query_eia(state, energy_source)` - Query EIA database
5. `generate_report(data, type)` - Generate formatted report

**Context**:
{context}

Plan and execute the workflow step-by-step:
1. First, outline the steps needed
2. Then execute each step
3. Finally, synthesize the results""",
    required_fields=["task_description", "context"]
)


# =============================================================================
# Report Templates
# =============================================================================

REPORT_TEMPLATES = {
    "executive_summary": """# Executive Summary

## Asset Overview
{asset_overview}

## Key Findings
{key_findings}

## Financial Highlights
{financial_highlights}

## Risk Assessment
{risk_assessment}

## Recommendation
{recommendation}

---
*Report generated by OpenEnergy Engine*
*Date: {date}*
""",

    "full_valuation": """# Asset Valuation Report

## 1. Executive Summary
{executive_summary}

## 2. Asset Description
{asset_description}

## 3. Market Analysis
{market_analysis}

## 4. Financial Analysis

### 4.1 Revenue Projections
{revenue_projections}

### 4.2 Cost Analysis
{cost_analysis}

### 4.3 Key Metrics
| Metric | Value |
|--------|-------|
| NPV | ${npv:,.0f} |
| IRR | {irr:.1%} |
| LCOE | ${lcoe:.2f}/MWh |
| Payback | {payback:.1f} years |

## 5. Risk Assessment
{risk_assessment}

## 6. Climate Analysis
{climate_analysis}

## 7. Conclusions & Recommendations
{conclusions}

---
*Prepared by OpenEnergy Engine*
*Valuation Date: {date}*
""",

    "climate_report": """# Climate Risk Assessment Report

## Location Profile
{location_profile}

## Climate Scenario: {scenario}

## Risk Summary
- **Overall Risk Score**: {risk_score:.2f}/1.0
- **Risk Category**: {risk_category}

## Extreme Event Analysis
{extreme_event_analysis}

## Resource Assessment
{resource_assessment}

## Long-term Projections
{projections}

## Adaptation Recommendations
{recommendations}

---
*Climate assessment by OpenEnergy Engine*
*Assessment Date: {date}*
"""
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_system_prompt(persona: str = "default") -> str:
    """Get system prompt for a persona."""
    return SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["default"])


def format_extreme_events(events: Dict[str, float]) -> str:
    """Format extreme event probabilities for prompt inclusion."""
    lines = []
    for event, prob in events.items():
        risk_level = "High" if prob > 0.3 else "Medium" if prob > 0.1 else "Low"
        lines.append(f"- {event.replace('_', ' ').title()}: {prob:.1%} ({risk_level})")
    return "\n".join(lines)


def format_asset_summary(asset: Dict[str, Any], index: int) -> str:
    """Format asset data for comparison prompt."""
    return f"""### Asset {index + 1}: {asset.get('asset_id', 'Unknown')}
- Type: {asset.get('asset_type', 'N/A')}
- Location: {asset.get('state', 'N/A')}
- Capacity: {asset.get('capacity_mw', 0):.1f} MW
- NPV: ${asset.get('npv', 0):,.0f}
- IRR: {asset.get('irr', 0):.1%}
- Risk Score: {asset.get('risk_score', 0):.2f}
"""


def build_analysis_prompt(
    data: Dict[str, Any],
    prompt_type: PromptType,
    **kwargs
) -> str:
    """Build a complete analysis prompt from data.

    Args:
        data: Source data dictionary
        prompt_type: Type of analysis prompt
        **kwargs: Additional template variables

    Returns:
        Formatted prompt string
    """
    templates = {
        PromptType.ASSET_ANALYSIS: ASSET_ANALYSIS_PROMPT,
        PromptType.CLIMATE_RISK: CLIMATE_RISK_PROMPT,
        PromptType.DETECTION: DETECTION_PROMPT,
        PromptType.COMPARISON: COMPARISON_PROMPT,
        PromptType.WORKFLOW: WORKFLOW_PROMPT,
    }

    template = templates.get(prompt_type)
    if not template:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    # Merge data and kwargs
    context = {**data, **kwargs}

    return template.format(**context)
