from __future__ import annotations

import json
import logging
import os
import re
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)

CONFIDENCE_RULE_MATCH = 1.0
CONFIDENCE_HEURISTIC = 0.7
CONFIDENCE_LLM_FALLBACK = 0.0
LLM_CALL_THRESHOLD = 0.7


# ── Shared helpers ────────────────────────────────────────────────────────────

def _flat(text: str) -> str:
    """Collapse all whitespace/newlines into single spaces for regex matching."""
    return re.sub(r"\s+", " ", text)


def _extract_transport_date(text: str) -> Tuple[Optional[str], float]:
    """
    Extract date from transport documents.
    Handles: JUN. 28, 2015 / JUN. 28,2015 / 28-JUN-2015 / 2015-06-28
    """
    flat = _flat(text)
    patterns = [
        # "JUN. 28, 2015" or "JUN. 28,2015"
        r"(?:dated?\s*at\s*[\w\s,]+?|on\s*board\s*date\s*:?\s*|date\s*of\s*issue\s*:?\s*|issued?\s*:?\s*)"
        r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})",
        # "28 JUN 2015" or "28-JUN-2015"
        r"(\d{1,2}[\s\-\.][A-Za-z]{3,9}[\s\-\.]\d{4})",
        # ISO "2015-06-28"
        r"(\d{4}-\d{2}-\d{2})",
        # "06/28/2015" or "28/06/2015"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})",
        # Standalone "JUN. 28, 2015" without label
        r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})",
    ]
    for pattern in patterns:
        m = re.search(pattern, flat, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            try:
                parsed = dateutil_parser.parse(raw, dayfirst=False)
                return parsed.date().isoformat(), CONFIDENCE_RULE_MATCH
            except Exception:
                return raw, CONFIDENCE_HEURISTIC
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_weight(text: str) -> Tuple[Optional[str], float]:
    """Extract gross weight value with unit."""
    flat = _flat(text)
    m = re.search(
        r"(?:gross\s*weight\s*:?\s*)?([\d,]+\.?\d*)\s*(KGS?|LBS?|MT|T)\b",
        flat, re.IGNORECASE
    )
    if m:
        return f"{m.group(1).replace(',', '')} {m.group(2).upper()}", CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_measurement(text: str) -> Tuple[Optional[str], float]:
    """Extract volume/measurement value with unit."""
    flat = _flat(text)
    m = re.search(
        r"([\d,]+\.?\d*)\s*(CBM|M3|CUB\.?\s*M|CFT)\b",
        flat, re.IGNORECASE
    )
    if m:
        return f"{m.group(1).replace(',', '')} {m.group(2).upper()}", CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_bl_number(text: str) -> Tuple[Optional[str], float]:
    """
    Extract B/L number. Looks for carrier-prefix + alphanumeric sequences
    (e.g. HDMU BUGA9183930, HLCU123456789) rather than matching the label text.
    """
    flat = _flat(text)
    # Carrier prefix (4 alpha) followed by alphanumeric BL number
    m = re.search(r"\b([A-Z]{4}[A-Z0-9]{6,12})\b", flat)
    if m:
        val = m.group(1)
        # Skip tokens that are clearly not BL numbers
        skip = {"LOAD", "COUNT", "WEIGHT", "CASE", "FREIGHT", "COPY"}
        if val.upper() not in skip and any(c.isdigit() for c in val):
            return val, CONFIDENCE_RULE_MATCH
    # Fallback: look explicitly after "B/L No" label
    m = re.search(r"B/L\s*No\.?\s+(?:[A-Z]{4}\s+)?([A-Z0-9]{6,15})", flat, re.IGNORECASE)
    if m:
        return m.group(1).strip(), CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_booking_no(text: str) -> Tuple[Optional[str], float]:
    """Extract booking number — typically alphanumeric, appears near 'Booking No'."""
    flat = _flat(text)
    m = re.search(r"Booking\s*No\.?\s*([A-Z0-9]{6,15})", flat, re.IGNORECASE)
    if m:
        return m.group(1).strip(), CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_container_refs(text: str) -> List[str]:
    """Extract container numbers (e.g. TCLU6016701, TRIU0663644)."""
    # Container numbers: 4 alpha + 7 digits
    return list(set(re.findall(r"\b([A-Z]{4}\d{7})\b", text)))


def _llm_extract_transport(
    text: str, missing_fields: List[str], doc_type: str
) -> Dict[str, Any]:
    """
    LLM fallback for transport document fields.
    Uses JSON mode and a detailed system prompt to handle form-layout documents
    where label-based regex fails.
    """
    if not missing_fields:
        return {}

    fields_desc = ", ".join(missing_fields)
    logger.info("LLM transport fallback — requesting: %s", fields_desc)

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        system = (
            f"You are a {doc_type} document extraction specialist. "
            "This document is a shipping form where field labels and their values "
            "may appear on separate lines or in adjacent table cells. "
            "You must extract ACTUAL VALUES, not field label names or template placeholder text. "
            "Rules:\n"
            "- Ignore template placeholder text like '(complete name and address)', "
            "'Booking No.', '/Voyage/Flag', 'For Transshipment to', 'Onward Inland Routing' "
            "— these are form labels, not values.\n"
            "- shipper: the actual company name and address of who shipped the goods\n"
            "- consignee: the actual company name and address of who receives the goods\n"
            "- vessel_name: the actual ship name (e.g. 'HYUNDAI FORWARD V#045E')\n"
            "- port_of_loading: actual departure port city/country (e.g. 'BUSAN, KOREA')\n"
            "- port_of_discharge: actual arrival port (e.g. 'SAVANNAH, GA')\n"
            "- description_of_goods: what cargo is being shipped\n"
            "- dates → ISO 8601 (YYYY-MM-DD)\n"
            "- null for any field not found\n"
            "You MUST respond with a single JSON object only."
        )

        human = (
            f"Document text:\n\n{text[:4000]}\n\n"
            f"Extract these fields as JSON: {fields_desc}"
        )

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = response.content.strip()
        result = json.loads(content)
        logger.info("LLM transport extracted: %s",
                    {k: v for k, v in result.items() if v is not None})
        return result

    except Exception as exc:
        logger.error("LLM transport extraction failed: %s", exc, exc_info=True)
        return {}


# ── Seaway Bill extraction node ───────────────────────────────────────────────

def extract_seaway_node(state: IDPState) -> IDPState:
    """
    LangGraph node — Seaway Bill / Bill of Lading Extraction.

    Strategy: seaway bills are form-layout documents where label-based regex
    is unreliable (labels and values appear on adjacent lines/cells).
    Rule-based is only used for fields with unambiguous value formats
    (BL number pattern, dates, weights, container numbers).
    All other fields default to LLM fallback.
    """
    _t0 = _time.perf_counter()

    parts = [t for t in [state.get("docling_output"), state.get("llamaparse_output")] if t]
    text = "\n\n---\n\n".join(parts) if parts else ""

    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    method_log: List[str] = []

    # ── Fields with reliable rule-based patterns ──────────────────────────────

    # BL number — carrier-prefix + digits format
    val, conf = _extract_bl_number(text)
    extracted["bl_number"] = val
    confidence["bl_number"] = conf
    method_log.append(f"bl_number: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Date of issue — specific date formats
    date_val, conf = _extract_transport_date(text)
    extracted["date_of_issue"] = date_val
    confidence["date_of_issue"] = conf
    method_log.append(f"date_of_issue: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Gross weight — numeric + unit
    val, conf = _extract_weight(text)
    extracted["gross_weight"] = val
    confidence["gross_weight"] = conf
    method_log.append(f"gross_weight: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Measurement / volume
    val, conf = _extract_measurement(text)
    extracted["measurement"] = val
    confidence["measurement"] = conf
    method_log.append(f"measurement: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # ── Fields that need LLM — form layout makes regex unreliable ─────────────
    for field in ["shipper", "consignee", "notify_party", "vessel_name",
                  "voyage_number", "port_of_loading", "port_of_discharge",
                  "place_of_delivery", "description_of_goods"]:
        extracted[field] = None
        confidence[field] = CONFIDENCE_LLM_FALLBACK
        method_log.append(f"{field}: pending_llm")

    # ── Reference IDs — container numbers + BL number ────────────────────────
    containers = _extract_container_refs(text)
    refs = containers[:]
    if extracted.get("bl_number") and extracted["bl_number"] not in refs:
        refs.insert(0, extracted["bl_number"])
    booking_val, _ = _extract_booking_no(text)
    if booking_val and booking_val not in refs:
        refs.append(booking_val)
    extracted["reference_ids"] = refs
    confidence["reference_ids"] = CONFIDENCE_RULE_MATCH if refs else CONFIDENCE_LLM_FALLBACK
    method_log.append(f"reference_ids: {'rule_based' if refs else 'pending_llm'} ({len(refs)} found)")

    # ── LLM fallback for low-confidence fields ────────────────────────────────
    missing = [f for f, s in confidence.items() if s < LLM_CALL_THRESHOLD]
    if missing:
        llm_results = _llm_extract_transport(text, missing, "sea waybill / bill of lading")
        for field, value in llm_results.items():
            if value is not None and extracted.get(field) is None:
                extracted[field] = value
                confidence[field] = 0.85
                method_log = [
                    log.replace(f"{field}: pending_llm", f"{field}: llm_fallback")
                    for log in method_log
                ]

    method_log = [log.replace("pending_llm", "not_found") for log in method_log]

    used_llm = any("llm_fallback" in m for m in method_log)
    source = "+".join(filter(None, [
        "docling" if state.get("docling_output") else None,
        "llamaparse" if state.get("llamaparse_output") else None,
    ]))
    extracted["extraction_method"] = f"{source}+{'rule_based+llm_fallback' if used_llm else 'rule_based'}"

    # Record timing
    timings: dict = state.get("node_timings") or {}
    timings["extract"] = round((_time.perf_counter() - _t0) * 1000, 2)

    return {
        **state,
        "extracted_fields": extracted,
        "confidence_scores": confidence,
        "extraction_method_log": method_log,
        "node_timings": timings,
    }


# ── Airway Bill extraction node ───────────────────────────────────────────────

def extract_airway_node(state: IDPState) -> IDPState:
    """LangGraph node — Air Waybill Extraction. Same strategy as seaway bill."""
    _t0 = _time.perf_counter()

    parts = [t for t in [state.get("docling_output"), state.get("llamaparse_output")] if t]
    text = "\n\n---\n\n".join(parts) if parts else ""

    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    method_log: List[str] = []

    # AWB number — specific numeric format
    flat = _flat(text)
    awb_m = re.search(r"\b(\d{3}-\d{8})\b", flat)  # standard AWB format: 123-12345678
    if not awb_m:
        awb_m = re.search(
            r"(?:awb|air\s*waybill)\s*(?:no|number|#)?\s*:?\s*([A-Z0-9\-]{6,20})",
            flat, re.IGNORECASE
        )
    extracted["awb_number"] = awb_m.group(1).strip() if awb_m else None
    confidence["awb_number"] = CONFIDENCE_RULE_MATCH if awb_m else CONFIDENCE_LLM_FALLBACK
    method_log.append(f"awb_number: {'rule_based' if awb_m else 'pending_llm'}")

    # Date
    date_val, conf = _extract_transport_date(text)
    extracted["date_of_issue"] = date_val
    confidence["date_of_issue"] = conf
    method_log.append(f"date_of_issue: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Weight and measurement
    val, conf = _extract_weight(text)
    extracted["gross_weight"] = val
    confidence["gross_weight"] = conf
    method_log.append(f"gross_weight: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    val, conf = _extract_measurement(text)
    extracted["measurement"] = val
    confidence["measurement"] = conf
    method_log.append(f"measurement: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # LLM-handled fields
    for field in ["shipper", "consignee", "issuing_carrier",
                  "airport_of_departure", "airport_of_destination",
                  "flight_number", "description_of_goods"]:
        extracted[field] = None
        confidence[field] = CONFIDENCE_LLM_FALLBACK
        method_log.append(f"{field}: pending_llm")

    # Reference IDs
    refs = []
    if extracted.get("awb_number"):
        refs.append(extracted["awb_number"])
    extracted["reference_ids"] = refs
    confidence["reference_ids"] = CONFIDENCE_RULE_MATCH if refs else CONFIDENCE_LLM_FALLBACK
    method_log.append(f"reference_ids: {'rule_based' if refs else 'pending_llm'}")

    # LLM fallback
    missing = [f for f, s in confidence.items() if s < LLM_CALL_THRESHOLD]
    if missing:
        llm_results = _llm_extract_transport(text, missing, "air waybill")
        for field, value in llm_results.items():
            if value is not None and extracted.get(field) is None:
                extracted[field] = value
                confidence[field] = 0.85
                method_log = [
                    log.replace(f"{field}: pending_llm", f"{field}: llm_fallback")
                    for log in method_log
                ]

    method_log = [log.replace("pending_llm", "not_found") for log in method_log]

    used_llm = any("llm_fallback" in m for m in method_log)
    source = "+".join(filter(None, [
        "docling" if state.get("docling_output") else None,
        "llamaparse" if state.get("llamaparse_output") else None,
    ]))
    extracted["extraction_method"] = f"{source}+{'rule_based+llm_fallback' if used_llm else 'rule_based'}"

    timings: dict = state.get("node_timings") or {}
    timings["extract"] = round((_time.perf_counter() - _t0) * 1000, 2)

    return {
        **state,
        "extracted_fields": extracted,
        "confidence_scores": confidence,
        "extraction_method_log": method_log,
        "node_timings": timings,
    }
