"""LangGraph ReAct Autonomous Agent for Astronomical Object Verification.

Autonomously reasons over AstroFoundation multi-modal neural network outputs
and queries SIMBAD, NED, and NASA ADS databases to synthesize astrophysical verdicts.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .schemas import (
    AdsPaper,
    AdsResult,
    AgentOutput,
    AgentReasoningStep,
    NedResult,
    SimbadResult,
)
from .tools import query_ads, query_ned, query_simbad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Graph State
# ---------------------------------------------------------------------------

class AstroAgentState(TypedDict):
    """Execution state tracked across the LangGraph reasoning cycle."""
    target_id: str
    kic_id: Optional[int]
    ra: Optional[float]
    dec: Optional[float]
    initial_class: str
    confidence: float
    energy_score: float
    is_anomaly: bool
    queried_tools: List[str]
    simbad_result: Optional[Dict[str, Any]]
    ned_result: Optional[Dict[str, Any]]
    ads_result: Optional[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    current_iteration: int
    next_action: str
    final_class: Optional[str]
    final_confidence: Optional[float]
    consensus_status: Optional[str]
    astronomical_notes: Optional[str]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def reason_node(state: AstroAgentState) -> Dict[str, Any]:
    """
    Evaluates current evidence and decides the next investigation step.
    Implements astrophysics decision heuristics matching survey protocols.
    """
    iteration = state.get("current_iteration", 0) + 1
    queried = set(state.get("queried_tools", []))
    init_class = state["initial_class"]
    conf = state["confidence"]
    is_anomaly = state["is_anomaly"]
    steps = list(state.get("reasoning_steps", []))

    thought: str = ""
    next_action: str = "synthesize"

    # Max iteration safety guard
    if iteration > 5:
        thought = "Maximum reasoning iterations (5) reached. Proceeding to final synthesis."
        next_action = "synthesize"
    elif is_anomaly:
        # Anomaly case: OOD energy flag requires exhaustive multi-catalog cross-match
        if "simbad" not in queried:
            thought = (
                f"Target flagged as potential ANOMALY (energy={state['energy_score']:.2f}). "
                "Querying SIMBAD first to check for any existing stellar or solar-system catalog entries."
            )
            next_action = "simbad"
        elif "ned" not in queried:
            thought = (
                "SIMBAD lookup complete. Target may be an extragalactic transient or host galaxy. "
                "Querying NED (NASA/IPAC Extragalactic Database)."
            )
            next_action = "ned"
        elif "ads" not in queried:
            thought = (
                "Catalog lookups concluded. Searching NASA ADS literature for prior studies or anomaly reports."
            )
            next_action = "ads"
        else:
            thought = "All databases cross-referenced for anomalous target. Synthesizing consensus."
            next_action = "synthesize"

    elif init_class in ["supernova", "gamma_ray_burst"]:
        # Explosive / extragalactic transients benefit from NED + ADS
        if "simbad" not in queried:
            thought = f"Target classified as {init_class}. Verifying local coordinates in SIMBAD."
            next_action = "simbad"
        elif "ned" not in queried:
            thought = f"Extragalactic transient ({init_class}) requires redshift check in NED."
            next_action = "ned"
        elif "ads" not in queried and conf < 0.90:
            thought = "Checking NASA ADS for published transient alerts or circulars."
            next_action = "ads"
        else:
            thought = "Sufficient catalog and literature evidence accumulated. Synthesizing verdict."
            next_action = "synthesize"

    elif conf >= 0.85:
        # High confidence stellar source (e.g. exoplanet transit, eclipsing binary, variable star)
        if "simbad" not in queried:
            thought = (
                f"Model predicts {init_class} with high confidence ({conf*100:.1f}%). "
                "Querying SIMBAD to cross-reference known spectral type and object classification."
            )
            next_action = "simbad"
        else:
            thought = "SIMBAD cross-match verified. No further database calls required for high-confidence source."
            next_action = "synthesize"

    else:
        # Ambiguous / low-confidence stellar source (0.50 <= conf < 0.85)
        if "simbad" not in queried:
            thought = (
                f"Classification {init_class} has moderate confidence ({conf*100:.1f}%). "
                "Querying SIMBAD for catalog ground truth."
            )
            next_action = "simbad"
        elif "ads" not in queried:
            thought = "Querying NASA ADS to resolve classification ambiguity with literature."
            next_action = "ads"
        else:
            thought = "Reasoning complete after multi-source consultation. Synthesizing results."
            next_action = "synthesize"

    steps.append({
        "step_index": iteration,
        "thought": thought,
        "action": next_action,
        "observation": f"Routing to node '{next_action}'"
    })

    return {
        "current_iteration": iteration,
        "next_action": next_action,
        "reasoning_steps": steps,
    }


def simbad_node(state: AstroAgentState) -> Dict[str, Any]:
    """Execute SIMBAD database lookup tool."""
    target_id = state["target_id"]
    ra = state.get("ra")
    dec = state.get("dec")

    result = query_simbad(target_id=target_id, ra=ra, dec=dec)
    queried = list(state.get("queried_tools", [])) + ["simbad"]

    obs = (
        f"SIMBAD found: {result.object_type} ({result.description or 'No desc'})"
        if result.found
        else f"SIMBAD: No entry found for {target_id}"
    )

    steps = list(state.get("reasoning_steps", []))
    if steps:
        steps[-1]["observation"] = obs

    return {
        "queried_tools": queried,
        "simbad_result": result.model_dump(),
        "reasoning_steps": steps,
    }


def ned_node(state: AstroAgentState) -> Dict[str, Any]:
    """Execute NED extragalactic database lookup tool."""
    target_id = state["target_id"]
    ra = state.get("ra")
    dec = state.get("dec")

    result = query_ned(target_id=target_id, ra=ra, dec=dec)
    queried = list(state.get("queried_tools", [])) + ["ned"]

    obs = (
        f"NED found: {result.object_type}, z={result.redshift}"
        if result.found
        else f"NED: Not indexed as extragalactic source"
    )

    steps = list(state.get("reasoning_steps", []))
    if steps:
        steps[-1]["observation"] = obs

    return {
        "queried_tools": queried,
        "ned_result": result.model_dump(),
        "reasoning_steps": steps,
    }


def ads_node(state: AstroAgentState) -> Dict[str, Any]:
    """Execute NASA ADS literature search tool."""
    target_id = state["target_id"]
    result = query_ads(query_term=target_id, target_name=target_id)
    queried = list(state.get("queried_tools", [])) + ["ads"]

    obs = (
        f"NASA ADS: Retrieved {len(result.papers)} papers ({result.literature_citations_count} citations total)"
        if result.found
        else f"NASA ADS: No relevant peer-reviewed papers for {target_id}"
    )

    steps = list(state.get("reasoning_steps", []))
    if steps:
        steps[-1]["observation"] = obs

    return {
        "queried_tools": queried,
        "ads_result": result.model_dump(),
        "reasoning_steps": steps,
    }


def synthesize_node(state: AstroAgentState) -> Dict[str, Any]:
    """
    Synthesize multi-modal network predictions with all cross-matched database evidence.
    Computes consensus status, adjusts final confidence, and builds scientific notes.
    """
    init_class = state["initial_class"]
    conf = state["confidence"]
    is_anomaly = state["is_anomaly"]
    simbad_dict = state.get("simbad_result") or {}
    ned_dict = state.get("ned_result") or {}
    ads_dict = state.get("ads_result") or {}

    final_class = init_class
    final_conf = conf
    consensus = "NO_PRIOR_CATALOG"
    notes_list: List[str] = []

    simbad_found = simbad_dict.get("found", False)
    simbad_type = (simbad_dict.get("object_type") or "").lower()

    if is_anomaly:
        if not simbad_found and not ned_dict.get("found", False):
            consensus = "NO_PRIOR_CATALOG"
            final_class = "anomaly"
            final_conf = max(conf, 0.92)
            notes_list.append(
                "Genuine novel candidate: Model flagged high OOD energy with zero prior catalog entries in SIMBAD or NED."
            )
        else:
            consensus = "REFINED"
            final_class = "anomaly"
            notes_list.append(
                f"Anomalous target indexed with known type '{simbad_dict.get('object_type')}'; "
                "exhibits atypical multi-modal signature warranting manual spectroscopic scrutiny."
            )

    elif simbad_found:
        # Check alignment between SIMBAD catalog type and model prediction
        matches = False
        inferred_simbad_class = None
        if "supernova" in simbad_type or "sn ia" in simbad_type or "sn ii" in simbad_type or "sn ib" in simbad_type:
            inferred_simbad_class = "supernova"
        elif "cataclysmic" in simbad_type or "dwarf nova" in simbad_type or "dn*" in simbad_type or "cv*" in simbad_type or "u gem" in simbad_type:
            inferred_simbad_class = "cataclysmic_variable"
        elif "cepheid" in simbad_type or "dcep" in simbad_type or "puls" in simbad_type or "delta sct" in simbad_type or "variable star" in simbad_type or "v*" in simbad_type:
            inferred_simbad_class = "variable_star"
        elif "eb*" in simbad_type or "binary" in simbad_type or "ecl" in simbad_type or "algol" in simbad_type:
            inferred_simbad_class = "eclipsing_binary"
        elif "pl?" in simbad_type or "planet" in simbad_type or "transit" in simbad_type:
            inferred_simbad_class = "exoplanet_transit"
        elif "grb" in simbad_type or "kilonova" in simbad_type or "gamma" in simbad_type or "merger" in simbad_type or "galaxy" in simbad_type:
            inferred_simbad_class = "gamma_ray_burst"

        if inferred_simbad_class and inferred_simbad_class == init_class:
            matches = True

        if matches:
            consensus = "CONFIRMED"
            final_conf = min(0.99, max(conf + 0.05, 0.92))
            notes_list.append(f"Confirmed by SIMBAD catalog object classification '{simbad_dict.get('object_type')}'.")
        elif inferred_simbad_class and conf < 0.85:
            # Autonomous Agent reconciliation: Ground-truth astronomical catalog overrides uncertain vision inference
            consensus = "RESOLVED_BY_AGENT"
            final_class = inferred_simbad_class
            final_conf = 0.945
            notes_list.append(
                f"Multi-modal neural classification was uncertain ({conf*100:.1f}% as {init_class}); "
                f"Agent autonomously cross-matched SIMBAD ground-truth catalog '{simbad_dict.get('object_type')}' "
                f"and resolved classification to '{inferred_simbad_class}'."
            )
        else:
            consensus = "AMBIGUOUS"
            notes_list.append(
                f"SIMBAD classifies object as '{simbad_dict.get('object_type')}', while model predicts '{init_class}'."
            )

    else:
        consensus = "NO_PRIOR_CATALOG"
        notes_list.append("Target not found in SIMBAD database; classification relies solely on AstroFoundation neural pipeline.")

    if ads_dict.get("found", False):
        paper_cnt = len(ads_dict.get("papers", []))
        cit_cnt = ads_dict.get("literature_citations_count", 0)
        notes_list.append(f"Literature corroboration: {paper_cnt} publications in NASA ADS ({cit_cnt} total citations).")

    notes = " ".join(notes_list)

    return {
        "final_class": final_class,
        "final_confidence": round(final_conf, 4),
        "consensus_status": consensus,
        "astronomical_notes": notes,
        "next_action": "finish",
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------

def route_next_step(state: AstroAgentState) -> str:
    """Determine the next graph execution branch from reason_node output."""
    action = state.get("next_action", "synthesize")
    if action == "simbad":
        return "simbad"
    elif action == "ned":
        return "ned"
    elif action == "ads":
        return "ads"
    elif action == "synthesize":
        return "synthesize"
    return END


# ---------------------------------------------------------------------------
# LangGraph Workflow Construction
# ---------------------------------------------------------------------------

def build_astro_agent_graph():
    """Build and compile the LangGraph ReAct agent state machine."""
    workflow = StateGraph(AstroAgentState)

    # Add core nodes
    workflow.add_node("reason", reason_node)
    workflow.add_node("simbad", simbad_node)
    workflow.add_node("ned", ned_node)
    workflow.add_node("ads", ads_node)
    workflow.add_node("synthesize", synthesize_node)

    # Set starting node
    workflow.set_entry_point("reason")

    # Add conditional branching from reason_node
    workflow.add_conditional_edges(
        "reason",
        route_next_step,
        {
            "simbad": "simbad",
            "ned": "ned",
            "ads": "ads",
            "synthesize": "synthesize",
            END: END,
        },
    )

    # Tool nodes cycle back to reason_node
    workflow.add_edge("simbad", "reason")
    workflow.add_edge("ned", "reason")
    workflow.add_edge("ads", "reason")

    # Final synthesis transitions to END
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# Global compiled app
astro_agent_app = build_astro_agent_graph()


def run_astro_agent(
    target_id: str,
    initial_class: str,
    confidence: float,
    energy_score: float = -0.5,
    is_anomaly: bool = False,
    kic_id: Optional[int] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
) -> AgentOutput:
    """
    Execute the LangGraph autonomous astronomical agent for a target.

    Args:
        target_id: Primary star or transient ID (e.g. 'Kepler-10', 'KIC 9002278')
        initial_class: Model predicted class name
        confidence: Model softmax confidence (0.0 to 1.0)
        energy_score: Negative logsumexp OOD energy score
        is_anomaly: Boolean flag indicating out-of-distribution detection
        kic_id: Optional integer Kepler Input Catalog ID
        ra: Optional Right Ascension
        dec: Optional Declination

    Returns:
        Validated AgentOutput Pydantic container with findings and reasoning trace.
    """
    initial_state: AstroAgentState = {
        "target_id": target_id,
        "kic_id": kic_id,
        "ra": ra,
        "dec": dec,
        "initial_class": initial_class,
        "confidence": confidence,
        "energy_score": energy_score,
        "is_anomaly": is_anomaly,
        "queried_tools": [],
        "simbad_result": None,
        "ned_result": None,
        "ads_result": None,
        "reasoning_steps": [],
        "current_iteration": 0,
        "next_action": "reason",
        "final_class": None,
        "final_confidence": None,
        "consensus_status": None,
        "astronomical_notes": None,
    }

    result = astro_agent_app.invoke(initial_state)

    # Convert trace to typed reasoning steps
    trace = [
        AgentReasoningStep(
            step_index=s["step_index"],
            thought=s["thought"],
            action=s["action"],
            observation=s["observation"],
        )
        for s in result.get("reasoning_steps", [])
    ]

    simbad_match = SimbadResult(**result["simbad_result"]) if result.get("simbad_result") else None
    ned_match = NedResult(**result["ned_result"]) if result.get("ned_result") else None
    ads_match = AdsResult(**result["ads_result"]) if result.get("ads_result") else None

    return AgentOutput(
        target_id=target_id,
        initial_class=initial_class,
        initial_confidence=confidence,
        energy_score=energy_score,
        is_anomaly=is_anomaly,
        catalogs_queried=result.get("queried_tools", []),
        simbad_match=simbad_match,
        ned_match=ned_match,
        ads_match=ads_match,
        reasoning_trace=trace,
        final_classification=result.get("final_class", initial_class),
        final_confidence=result.get("final_confidence", confidence),
        consensus_status=result.get("consensus_status", "NO_PRIOR_CATALOG"),
        astronomical_notes=result.get("astronomical_notes", ""),
    )
