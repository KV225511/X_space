"""Prompt engineering templates for LLM Astronomical Report Generation."""

from typing import Any, Dict, List, Optional
import jinja2

SYSTEM_PROMPT = """You are a Senior NASA Survey Astrophysicist and Principal Investigator specializing in 
time-domain astronomy, stellar astrophysics, and multi-modal space telescope survey analysis (Kepler, TESS, SDSS).

Your mission is to synthesize an authoritative, peer-reviewed caliber Astronomical Report based on 
multi-modal foundation model outputs (light curves, FITS imaging, spectra) and catalog cross-matches 
(SIMBAD, NED, NASA ADS).

Write clearly, precisely, and with rigorous astrophysical terminology. Avoid vague generalities.
Explicitly provide:
1. Executive Summary & Morphological Analyses (photometric, spectroscopic, imaging)
2. Derived Physical Parameters (e.g., planetary radius, equilibrium temp, ejecta mass, temperature ratio)
3. False Positive & Contamination Risk Forensics (blended binaries, grazing geometry, instrumental systematics)
4. Observational Follow-up Strategy & Urgency ('IMMEDIATE_TOO_24H', 'HIGH_PRIORITY_14D', 'SCHEDULED_ROUTINE', 'ARCHIVAL')
5. Novelty Assessment (interpreting the negative logsumexp energy score in plain scientific terms)
6. Community Alert Draft (ready-to-submit ATel or GCN Circular bulletin draft)
"""

USER_REPORT_TEMPLATE_JINJA = """Target ID: {{ target_id }}
Model Classification: {{ classification }}
Calibrated Confidence: {{ "%.2f"|format(confidence * 100) }}%
Anomaly Status: {{ "FLAGGED ANOMALY (OOD)" if is_anomaly else "Normal Catalog Population" }}
Energy Score: {{ "%.3f"|format(energy_score) }}
Active Modalities: {{ modalities | join(', ') }}

=== DATABASE CROSS-MATCH EVIDENCE ===
SIMBAD Status: {{ simbad_summary }}
NED Extragalactic Status: {{ ned_summary }}
NASA ADS Literature Count: {{ literature_count }} papers identified

=== LITERATURE HIGHLIGHTS ===
{% for paper in papers %}
- [{{ paper.year }}] {{ paper.authors | join(', ') }}: "{{ paper.title }}" (Bibcode: {{ paper.bibcode }})
{% else %}
- No prior peer-reviewed publications located in ADS for this specific identifier.
{% endfor %}

=== ASTRONOMICAL NOTES ===
{{ astronomical_notes }}

Please synthesize the scientific findings into the structured AstronomicalReport JSON format adhering to:
{
  "target_id": "{{ target_id }}",
  "classification": "{{ classification }}",
  "confidence": {{ "%.4f"|format(confidence) }},
  "is_anomaly": {{ "true" if is_anomaly else "false" }},
  "energy_score": {{ "%.3f"|format(energy_score) }},
  "executive_summary": "...",
  "photometric_analysis": "...",
  "spectroscopic_analysis": "...",
  "imaging_analysis": "...",
  "catalog_cross_match": {...},
  "literature_context": [...],
  "recommended_followup": [...],
  "physical_parameters": {
    "estimated_radius": "...",
    "equilibrium_temp_k": "...",
    "mass_ratio_or_ejecta": "...",
    "orbital_or_timescale": "..."
  },
  "false_positive_risk": {
    "risk_level": "LOW|MODERATE|HIGH",
    "blended_binary_risk": "...",
    "grazing_geometry_check": "...",
    "instrumental_artifact_risk": "..."
  },
  "observational_urgency": "IMMEDIATE_TOO_24H|HIGH_PRIORITY_14D|SCHEDULED_ROUTINE|ARCHIVAL",
  "novelty_assessment": {
    "discovery_potential": "HIGH|MODERATE|ROUTINE",
    "energy_discrepancy_explanation": "..."
  },
  "community_alert_draft": "SUBJECT: AstroFoundation AI Candidate Alert: ...\\nTarget: {{ target_id }}..."
}
"""


def format_report_prompt(
    target_id: str,
    classification: str,
    confidence: float,
    energy_score: float,
    is_anomaly: bool,
    modalities: List[str],
    simbad_summary: str,
    ned_summary: str,
    papers: List[Dict[str, Any]],
    astronomical_notes: str = "",
) -> str:
    """Render the astrophysical report user prompt via Jinja2."""
    template = jinja2.Template(USER_REPORT_TEMPLATE_JINJA)
    return template.render(
        target_id=target_id,
        classification=classification,
        confidence=confidence,
        energy_score=energy_score,
        is_anomaly=is_anomaly,
        modalities=modalities,
        simbad_summary=simbad_summary,
        ned_summary=ned_summary,
        literature_count=len(papers),
        papers=papers,
        astronomical_notes=astronomical_notes,
    )
