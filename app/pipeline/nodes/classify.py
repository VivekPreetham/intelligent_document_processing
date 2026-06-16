from __future__ import annotations

import logging
import os
import re

from app.pipeline.state import DocumentType, IDPState
from app.pipeline.timing import node_timer

logger = logging.getLogger(__name__)

# ── Keyword rules ─────────────────────────────────────────────────────────────
# Each document type has a set of strong signal keywords.
# We scan the first 1000 characters of the parsed text — document headers
# and titles always appear early, so this is sufficient and cheap.

_RULES: dict[DocumentType, list[str]] = {
    "invoice": [
        r"\binvoice\b",
        r"\binv\b",
        r"\btax invoice\b",
        r"\bproforma invoice\b",
        r"\bcommercial invoice\b",
        r"\bbill to\b",
        r"\bdue date\b",
        r"\bamount due\b",
    ],
    "seaway_bill": [
        r"\bsea waybill\b",
        r"\bbill of lading\b",
        r"\bb/l\b",
        r"\bbl number\b",
        r"\bport of loading\b",
        r"\bport of discharge\b",
        r"\bvessel\b",
        r"\bvoyage\b",
        r"\bshipper\b",
        r"\bconsignee\b",
    ],
    "airway_bill": [
        r"\bair waybill\b",
        r"\bairway bill\b",
        r"\bawb\b",
        r"\bairport of departure\b",
        r"\bairport of destination\b",
        r"\bflight\b",
        r"\bcarrier\b",
        r"\biata\b",
    ],
}


def _rule_based_classify(text: str) -> DocumentType | None:
    """
    Scan the first 1000 chars of the text for strong keyword signals.
    Returns a document type if a clear match is found, else None.
    We count keyword hits per type and return the type with the most hits,
    provided it has at least 2 — avoiding false positives from a single word.
    """
    sample = text[:1000].lower()
    scores: dict[DocumentType, int] = {t: 0 for t in _RULES}

    for doc_type, patterns in _RULES.items():
        for pattern in patterns:
            if re.search(pattern, sample):
                scores[doc_type] += 1

    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]

    if best_score >= 2:
        logger.info("Rule-based classification → %s (score=%d)", best_type, best_score)
        return best_type

    if best_score == 1:
        # Single hit is a weak signal — log it but still return it
        # rather than paying for an LLM call for obvious documents.
        logger.info(
            "Rule-based classification (weak) → %s (score=1)", best_type
        )
        return best_type

    return None


def _llm_classify(text: str) -> DocumentType:
    """
    Fall back to Groq LLM when keyword rules are inconclusive.
    We pass only the first 1500 chars to keep token usage minimal —
    classification does not need the full document body.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
        )

        system = (
            "You are a document classification assistant. "
            "Classify the document excerpt into exactly one of these categories: "
            "invoice, seaway_bill, airway_bill, unknown. "
            "Respond with only the category label — no explanation, no punctuation."
        )
        human = f"Document excerpt:\n\n{text[:1500]}"

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        label = response.content.strip().lower().replace(" ", "_")

        valid: list[DocumentType] = ["invoice", "seaway_bill", "airway_bill", "unknown"]
        if label in valid:
            logger.info("LLM classification → %s", label)
            return label  # type: ignore[return-value]

        logger.warning("LLM returned unexpected label '%s' — defaulting to unknown", label)
        return "unknown"

    except Exception as exc:
        logger.error("LLM classification failed: %s", exc)
        return "unknown"


def classify_node(state: IDPState) -> IDPState:
    """
    LangGraph node — Classify.

    Reads:   docling_output, llamaparse_output
    Writes:  document_type

    Strategy:
    1. Prefer docling output for classification (layout-aware, faster).
    2. Fall back to llamaparse output if docling is None.
    3. Run rule-based keyword scoring first — no LLM cost for clear cases.
    4. Only call the Groq LLM if keyword rules return no match.
    """
    docling_text: str | None = state.get("docling_output")
    llamaparse_text: str | None = state.get("llamaparse_output")

    # Use whichever parser succeeded; prefer docling.
    text = docling_text or llamaparse_text or ""

    if not text.strip():
        logger.error("No parsed text available for classification")
        return {**state, "document_type": "unknown"}

    with node_timer(state, "classify"):
        doc_type = _rule_based_classify(text)
        if doc_type is None:
            logger.info("Rule-based classification inconclusive — calling LLM")
            doc_type = _llm_classify(text)

    return {**state, "document_type": doc_type}
