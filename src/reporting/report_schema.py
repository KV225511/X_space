"""Pydantic Schema for AstroFoundation Publication-Style Astronomical Reports."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AstronomicalReport(BaseModel):
    """
    Validated scientific report schema for multi-modal astronomical event analysis.
    Adheres to NASA ADS / peer-reviewed astrophysical reporting guidelines.
    Includes astrophysical parameter derivations, contamination forensics,
    observational urgency, and community alert drafts.
    """
    target_id: str = Field(description="Primary catalog designation (e.g. Kepler-10, KIC 9002278)")
    classification: str = Field(description="Predicted or consensus astronomical class")
    confidence: float = Field(description="Calibrated posterior confidence (0.0 to 1.0)")
    is_anomaly: bool = Field(default=False, description="Whether flagged as out-of-distribution anomaly")
    energy_score: float = Field(description="Negative logsumexp OOD energy score")
    
    executive_summary: str = Field(description="High-level scientific summary for observational astronomers")
    photometric_analysis: str = Field(description="Light curve morphology and transit/variability details")
    spectroscopic_analysis: str = Field(description="Spectral features, continuum, and emission/absorption line context")
    imaging_analysis: str = Field(description="FITS spatial profile, PSF characteristics, and contamination check")
    
    catalog_cross_match: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cross-match findings from SIMBAD, NED, and Gaia catalogs"
    )
    literature_context: List[str] = Field(
        default_factory=list,
        description="Relevant peer-reviewed papers and bibcodes from NASA ADS"
    )
    recommended_followup: List[str] = Field(
        default_factory=list,
        description="Actionable telescope followup recommendations (e.g. high-resolution spectroscopy, RV)"
    )
    data_modalities_present: List[str] = Field(
        default_factory=list,
        description="List of modalities used: ['light_curve', 'fits_image', 'spectrum']"
    )

    # Enhanced Astrophysical Insights
    physical_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Derived physical parameters (e.g. estimated radius, equilibrium temperature, mass ratio, ejecta mass)",
    )
    false_positive_risk: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contamination analysis (e.g. blended binary risk, grazing geometry, instrumental artifact likelihood)",
    )
    observational_urgency: str = Field(
        default="SCHEDULED_ROUTINE",
        description="Observational follow-up priority: IMMEDIATE_TOO_24H, HIGH_PRIORITY_14D, SCHEDULED_ROUTINE, or ARCHIVAL",
    )
    novelty_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Plain-English OOD energy discrepancy and discovery potential metric",
    )
    community_alert_draft: Optional[str] = Field(
        default=None,
        description="Formatted Astronomer's Telegram (ATel) or GCN Circular bulletin draft",
    )

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of report generation"
    )
    model_version: str = Field(default="AstroFoundation-v4.0", description="Foundation model pipeline release")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata including DeepSeek source, cache status, and moderation metrics"
    )
