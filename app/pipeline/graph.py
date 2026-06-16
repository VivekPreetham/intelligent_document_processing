from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.pipeline.nodes.classify import classify_node
from app.pipeline.nodes.error_node import error_node
from app.pipeline.nodes.extract import extract_invoice_node
from app.pipeline.nodes.extract_transport import extract_airway_node, extract_seaway_node
from app.pipeline.nodes.parse import parse_node
from app.pipeline.nodes.validate import validate_node
from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)


# ── Routing functions (conditional edges) ────────────────────────────────────

def route_after_parse(state: IDPState) -> str:
    """
    After parse_node: if both parsers failed, short-circuit to error.
    Otherwise proceed to classification.
    """
    if state.get("is_fatal_error"):
        logger.info("Routing → error (fatal parse failure)")
        return "error"
    return "classify"


def route_after_classify(state: IDPState) -> str:
    """
    After classify_node: route to the correct extraction node based on
    document_type, or to error if the document is unrecognised.
    """
    doc_type = state.get("document_type", "unknown")
    routes = {
        "invoice": "extract_invoice",
        "seaway_bill": "extract_seaway",
        "airway_bill": "extract_airway",
    }
    destination = routes.get(doc_type, "error")
    logger.info("Routing → %s (document_type=%s)", destination, doc_type)
    return destination


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the IDP LangGraph pipeline.

    Node layout:
        parse
          ├─ (fatal error) ──────────────────────────► error → END
          └─ (ok) ──► classify
                        ├─ invoice    ──► extract_invoice  ──► validate → END
                        ├─ seaway_bill ─► extract_seaway   ──► validate → END
                        ├─ airway_bill ─► extract_airway   ──► validate → END
                        └─ unknown    ──────────────────────► error → END

    The graph is compiled once at startup and reused for every request,
    avoiding repeated graph construction overhead.
    """
    builder = StateGraph(IDPState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("parse", parse_node)
    builder.add_node("classify", classify_node)
    builder.add_node("extract_invoice", extract_invoice_node)
    builder.add_node("extract_seaway", extract_seaway_node)
    builder.add_node("extract_airway", extract_airway_node)
    builder.add_node("validate", validate_node)
    builder.add_node("error", error_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("parse")

    # ── Conditional edge: parse → classify | error ────────────────────────────
    builder.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "classify": "classify",
            "error": "error",
        },
    )

    # ── Conditional edge: classify → extraction node | error ──────────────────
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "extract_invoice": "extract_invoice",
            "extract_seaway": "extract_seaway",
            "extract_airway": "extract_airway",
            "error": "error",
        },
    )

    # ── All extraction nodes → validate ───────────────────────────────────────
    builder.add_edge("extract_invoice", "validate")
    builder.add_edge("extract_seaway", "validate")
    builder.add_edge("extract_airway", "validate")

    # ── Terminal edges ────────────────────────────────────────────────────────
    builder.add_edge("validate", END)
    builder.add_edge("error", END)

    return builder.compile()


# ── Singleton compiled graph — imported by FastAPI routes ────────────────────
# Built once when the module is first imported; reused for all requests.
idp_graph = build_graph()
