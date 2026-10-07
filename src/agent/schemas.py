"""Pydantic schemas for AstroFoundation LangGraph autonomous agent."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TargetCoordinates(BaseModel):
    """Sky coordinates for target astronomical object."""
    ra: Optional[float] = Field(default=None, description="Right Ascension in decimal degrees (J2000)")
    dec: Optional[float] = Field(default=None, description="Declination in decimal degrees (J2000)")


class CatalogResult(BaseModel):
    """Base schema for astronomical database cross-match results."""
    catalog_name: str
    target_id: str
    found: bool = False
    main_id: Optional[str] = None
    object_type: Optional[str] = None
    description: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    redshift: Optional[float] = None
    literature_citations_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimbadResult(CatalogResult):
    """SIMBAD Astronomical Database query result schema."""
    catalog_name: str = "SIMBAD"
    spectral_type: Optional[str] = None
    morphology: Optional[str] = None
    proper_motion_ra: Optional[float] = None
    proper_motion_dec: Optional[float] = None
    parallax_mas: Optional[float] = None


class NedResult(CatalogResult):
    """NASA/IPAC Extragalactic Database (NED) query result schema."""
    catalog_name: str = "NED"
    galaxy_morphology: Optional[str] = None
    heliocentric_redshift: Optional[float] = None
    velocity_kms: Optional[float] = None


class AdsPaper(BaseModel):
    """Bibliographic entry from NASA Astrophysics Data System (ADS)."""
    bibcode: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    citation_count: Optional[int] = 0
    abstract_snippet: Optional[str] = None


class AdsResult(CatalogResult):
    """NASA ADS literature query result schema."""
    catalog_name: str = "NASA_ADS"
    papers: List[AdsPaper] = Field(default_factory=list)


class AgentReasoningStep(BaseModel):
    """A single ReAct reasoning cycle step."""
    step_index: int
    thought: str
    action: str
    observation: str


class AgentOutput(BaseModel):
    """Final output emitted by the AstroFoundation autonomous LangGraph agent."""
    target_id: str
    initial_class: str
    initial_confidence: float
    energy_score: float
    is_anomaly: bool
    catalogs_queried: List[str] = Field(default_factory=list)
    simbad_match: Optional[SimbadResult] = None
    ned_match: Optional[NedResult] = None
    ads_match: Optional[AdsResult] = None
    reasoning_trace: List[AgentReasoningStep] = Field(default_factory=list)
    final_classification: str
    final_confidence: float
    consensus_status: str = Field(
        default="NO_PRIOR_CATALOG",
        description="One of: CONFIRMED, CONFLICT, REFINED, AMBIGUOUS, NO_PRIOR_CATALOG"
    )
    astronomical_notes: str = ""
