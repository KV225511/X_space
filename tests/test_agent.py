"""Unit tests for AstroFoundation LangGraph Agent and Astronomical Database Tools."""

import os
import pytest

from src.agent.agent import astro_agent_app, run_astro_agent
from src.agent.schemas import AgentOutput, AdsResult, NedResult, SimbadResult
from src.agent.tools import query_ads, query_ned, query_simbad


@pytest.fixture(autouse=True)
def setup_offline_environment():
    """Ensure tests run deterministically without hitting external network bottlenecks."""
    os.environ["ASTRO_OFFLINE_MODE"] = "1"
    yield
    os.environ.pop("ASTRO_OFFLINE_MODE", None)


def test_query_simbad_curated_target():
    """Verify SIMBAD query tool returns accurate metadata for Kepler-10."""
    res = query_simbad("Kepler-10")
    assert isinstance(res, SimbadResult)
    assert res.found is True
    assert "Planet" in (res.object_type or "") or "Pl" in (res.object_type or "")
    assert res.spectral_type == "G0V"
    assert res.ra is not None
    assert res.dec is not None


def test_query_ned_curated_supernova():
    """Verify NED query tool retrieves extragalactic redshift for ASASSN-15lh."""
    res = query_ned("ASASSN-15lh")
    assert isinstance(res, NedResult)
    assert res.found is True
    assert res.redshift == pytest.approx(0.2326, abs=1e-3)
    assert "Galaxy" in (res.object_type or "")


def test_query_ads_literature():
    """Verify NASA ADS tool retrieves published discovery papers."""
    res = query_ads("Kepler-10")
    assert isinstance(res, AdsResult)
    assert res.found is True
    assert len(res.papers) >= 1
    assert any("Batalha" in p.authors[0] for p in res.papers)


def test_agent_high_confidence_confirmation():
    """Verify agent workflow confirms high-confidence exoplanet transit via SIMBAD."""
    output = run_astro_agent(
        target_id="Kepler-10",
        initial_class="exoplanet_transit",
        confidence=0.95,
        energy_score=-0.2,
        is_anomaly=False,
    )
    assert isinstance(output, AgentOutput)
    assert output.final_classification == "exoplanet_transit"
    assert output.consensus_status == "CONFIRMED"
    assert "simbad" in output.catalogs_queried
    assert output.simbad_match is not None
    assert output.final_confidence >= 0.95


def test_agent_eclipsing_binary_confirmation():
    """Verify agent workflow on Kepler Eclipsing Binary catalog target."""
    output = run_astro_agent(
        target_id="KIC 9002278",
        initial_class="eclipsing_binary",
        confidence=0.91,
        energy_score=-0.3,
        is_anomaly=False,
    )
    assert output.final_classification == "eclipsing_binary"
    assert output.consensus_status == "CONFIRMED"
    assert "simbad" in output.catalogs_queried


def test_agent_anomaly_exhaustive_crossmatch():
    """Verify anomalous target triggers exhaustive multi-catalog search."""
    output = run_astro_agent(
        target_id="DeepSpace-Anomaly-X",
        initial_class="variable_star",
        confidence=0.42,
        energy_score=-2.85,
        is_anomaly=True,
    )
    assert output.is_anomaly is True
    assert output.final_classification == "anomaly"
    assert output.consensus_status in ("NO_PRIOR_CATALOG", "REFINED")
    # All three databases must have been cross-checked for an anomaly
    assert "simbad" in output.catalogs_queried
    assert "ned" in output.catalogs_queried
    assert "ads" in output.catalogs_queried


def test_agent_graph_compilation_and_trace():
    """Verify underlying LangGraph compiles and produces a step-by-step reasoning trace."""
    assert astro_agent_app is not None
    output = run_astro_agent(
        target_id="KIC-9700322",
        initial_class="variable_star",
        confidence=0.88,
    )
    assert len(output.reasoning_trace) >= 1
    assert all(hasattr(step, "thought") for step in output.reasoning_trace)
    assert all(hasattr(step, "action") for step in output.reasoning_trace)
