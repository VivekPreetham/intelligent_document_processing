from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from app.models.errors import ExtractionError


# Document type labels used for routing conditional edges in the graph.
DocumentType = Literal["invoice", "seaway_bill", "airway_bill", "unknown", "parse_failed"]


class IDPState(TypedDict, total=False):
    """
    Shared state that flows through every node in the LangGraph pipeline.

    Each node reads what it needs and writes its outputs back into this dict.
    LangGraph passes the same state object along every edge, so nodes
    communicate purely through this structure — no global variables, no
    side channels.

    Fields marked Optional are populated incrementally as the graph runs.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    raw_pdf_bytes: bytes
    # Original filename, used for logging and the extraction_method trail.
    filename: str

    # ── Parse node outputs ───────────────────────────────────────────────────
    # Markdown / structured text produced by docling (layout-aware).
    docling_output: Optional[str]
    # Markdown produced by LlamaParse (LLM-optimised).
    llamaparse_output: Optional[str]

    # ── Classify node output ─────────────────────────────────────────────────
    document_type: Optional[DocumentType]

    # ── Extraction node outputs ──────────────────────────────────────────────
    # Raw extracted values keyed by field name before Pydantic validation.
    extracted_fields: Optional[Dict[str, Any]]
    # Per-field confidence: 1.0 = clean regex match, 0.7 = heuristic,
    # 0.0 = not found, handed to LLM fallback.
    confidence_scores: Optional[Dict[str, float]]
    # Human-readable log of which path each field took through the pipeline,
    # e.g. ["date: rule_based", "issuer: llm_fallback", "total: docling_table"]
    extraction_method_log: Optional[List[str]]

    # ── Validation node outputs ──────────────────────────────────────────────
    # Accumulated field-level failures from the validation node.
    validation_errors: Optional[List[ExtractionError]]
    # The finished, Pydantic-validated response model (or None on total failure).
    final_response: Optional[Any]

    # ── Error flag ───────────────────────────────────────────────────────────
    # Set to True by the parse node if both parsers fail completely,
    # which routes the graph to the error exit node instead of extraction.
    is_fatal_error: Optional[bool]
    fatal_error_message: Optional[str]

    # ── Timing ───────────────────────────────────────────────────────────────
    # Each node writes its elapsed time (ms) here so the final response
    # can expose per-node and total processing times for observability.
    node_timings: Optional[Dict[str, float]]
