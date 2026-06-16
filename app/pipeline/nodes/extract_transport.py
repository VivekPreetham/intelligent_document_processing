from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)

CONFIDENCE_RULE_MATCH = 1.0
CONFIDENCE_HEURISTIC = 0.7
CONFIDENCE_LLM_FALLBACK = 0.0
LLM_CALL_THRESHOLD = 0.7


# ── Shared transport helpers ──────────────────────────────────────────────────

def _extract_field(text: str, patterns: List[str]) -> Tuple[Optional[str], float]:
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_transport_date(text: str) -> Tuple[Optional[str], float]:
    patterns = [
        r"(?:date\s*of\s*issue|issued|date)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date\s*of\s*issue|issued|date)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            try:
                parsed = dateutil_parser.parse(raw, dayfirst=False)
                return parsed.date().isoformat(), CONFIDENCE_RULE_MATCH
            except Exception:
                return raw, CONFIDENCE_HEURISTIC
    return None, CONFIDENCE_LLM_FALLBACK


def _llm_extract_transport(text: str, missing_fields: List[str], doc_type: str) -> Dict[str, Any]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        if not missing_fields:
            return {}

        fields_desc = ", ".join(missing_fields)
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
        )

        system = (
            f"You are a {doc_type} document extraction assistant. "
            f"Extract the following fields: {fields_desc}. "
            "Return dates in ISO 8601 format (YYYY-MM-DD). "
            "Return monetary amounts as plain numbers. "
            "Return null for any field not found in the text. "
            "Do not hallucinate values."
        )
        human = f"Document text:\n\n{text[:3000]}\n\nExtract: {fields_desc}"

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        return json.loads(content)

    except Exception as exc:
        logger.error("LLM transport extraction failed: %s", exc)
        return {}


# ── Seaway Bill extraction node ───────────────────────────────────────────────

def extract_seaway_node(state: IDPState) -> IDPState:
    """LangGraph node — Seaway Bill / Bill of Lading Extraction."""
    parts = [t for t in [state.get("docling_output"), state.get("llamaparse_output")] if t]
    text = "\n\n---\n\n".join(parts) if parts else ""
    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    method_log: List[str] = []

    # B/L number
    val, conf = _extract_field(text, [
        r"(?:b/l\s*no|bl\s*number|bill\s*of\s*lading\s*no)\s*:?\s*([A-Z0-9\-\/]+)",
        r"(?:b/l|bl)\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
    ])
    extracted["bl_number"] = val; confidence["bl_number"] = conf
    method_log.append(f"bl_number: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Shipper
    val, conf = _extract_field(text, [r"shipper\s*:?\s*([^\n]{3,80})"])
    extracted["shipper"] = val; confidence["shipper"] = conf
    method_log.append(f"shipper: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Consignee
    val, conf = _extract_field(text, [r"consignee\s*:?\s*([^\n]{3,80})"])
    extracted["consignee"] = val; confidence["consignee"] = conf
    method_log.append(f"consignee: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Vessel & voyage
    val, conf = _extract_field(text, [r"vessel\s*(?:name)?\s*:?\s*([^\n]{2,60})"])
    extracted["vessel_name"] = val; confidence["vessel_name"] = conf
    method_log.append(f"vessel_name: {'rule_based' if conf > 0 else 'pending_llm'}")

    val, conf = _extract_field(text, [r"voyage\s*(?:no|number)?\s*:?\s*([A-Z0-9\-\/]+)"])
    extracted["voyage_number"] = val; confidence["voyage_number"] = conf
    method_log.append(f"voyage_number: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Ports
    val, conf = _extract_field(text, [r"port\s*of\s*loading\s*:?\s*([^\n]{2,60})"])
    extracted["port_of_loading"] = val; confidence["port_of_loading"] = conf
    method_log.append(f"port_of_loading: {'rule_based' if conf > 0 else 'pending_llm'}")

    val, conf = _extract_field(text, [r"port\s*of\s*discharge\s*:?\s*([^\n]{2,60})"])
    extracted["port_of_discharge"] = val; confidence["port_of_discharge"] = conf
    method_log.append(f"port_of_discharge: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Date
    date_val, conf = _extract_transport_date(text)
    extracted["date_of_issue"] = date_val; confidence["date_of_issue"] = conf
    method_log.append(f"date_of_issue: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Reference IDs
    refs = []
    for m in re.finditer(r"(?:reference|ref)\s*#?\s*:?\s*([A-Z0-9\-\/]+)", text, re.IGNORECASE):
        refs.append(m.group(1).strip())
    if extracted.get("bl_number"):
        refs.insert(0, extracted["bl_number"])
    extracted["reference_ids"] = refs
    confidence["reference_ids"] = CONFIDENCE_RULE_MATCH if refs else CONFIDENCE_LLM_FALLBACK

    # LLM fallback
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

    return {**state, "extracted_fields": extracted, "confidence_scores": confidence, "extraction_method_log": method_log}


# ── Airway Bill extraction node ───────────────────────────────────────────────

def extract_airway_node(state: IDPState) -> IDPState:
    """LangGraph node — Air Waybill Extraction."""
    parts = [t for t in [state.get("docling_output"), state.get("llamaparse_output")] if t]
    text = "\n\n---\n\n".join(parts) if parts else ""
    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    method_log: List[str] = []

    # AWB number
    val, conf = _extract_field(text, [
        r"(?:awb|air\s*waybill)\s*(?:no|number|#)?\s*:?\s*([0-9\-]{8,20})",
        r"(?:airway\s*bill)\s*(?:no|number)?\s*:?\s*([A-Z0-9\-]+)",
    ])
    extracted["awb_number"] = val; confidence["awb_number"] = conf
    method_log.append(f"awb_number: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Shipper & Consignee
    val, conf = _extract_field(text, [r"shipper\s*:?\s*([^\n]{3,80})"])
    extracted["shipper"] = val; confidence["shipper"] = conf
    method_log.append(f"shipper: {'rule_based' if conf > 0 else 'pending_llm'}")

    val, conf = _extract_field(text, [r"consignee\s*:?\s*([^\n]{3,80})"])
    extracted["consignee"] = val; confidence["consignee"] = conf
    method_log.append(f"consignee: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Airports
    val, conf = _extract_field(text, [
        r"airport\s*of\s*departure\s*:?\s*([^\n]{2,60})",
        r"from\s*(?:airport)?\s*:?\s*([^\n]{2,60})",
    ])
    extracted["airport_of_departure"] = val; confidence["airport_of_departure"] = conf
    method_log.append(f"airport_of_departure: {'rule_based' if conf > 0 else 'pending_llm'}")

    val, conf = _extract_field(text, [
        r"airport\s*of\s*destination\s*:?\s*([^\n]{2,60})",
        r"to\s*(?:airport)?\s*:?\s*([^\n]{2,60})",
    ])
    extracted["airport_of_destination"] = val; confidence["airport_of_destination"] = conf
    method_log.append(f"airport_of_destination: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Flight
    val, conf = _extract_field(text, [r"flight\s*(?:no|number)?\s*:?\s*([A-Z0-9\-\/]+)"])
    extracted["flight_number"] = val; confidence["flight_number"] = conf
    method_log.append(f"flight_number: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Carrier
    val, conf = _extract_field(text, [r"(?:issuing\s*carrier|carrier)\s*:?\s*([^\n]{2,60})"])
    extracted["issuing_carrier"] = val; confidence["issuing_carrier"] = conf
    method_log.append(f"issuing_carrier: {'rule_based' if conf > 0 else 'pending_llm'}")

    # Date
    date_val, conf = _extract_transport_date(text)
    extracted["date_of_issue"] = date_val; confidence["date_of_issue"] = conf
    method_log.append(f"date_of_issue: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # Reference IDs
    refs = []
    if extracted.get("awb_number"):
        refs.append(extracted["awb_number"])
    extracted["reference_ids"] = refs
    confidence["reference_ids"] = CONFIDENCE_RULE_MATCH if refs else CONFIDENCE_LLM_FALLBACK

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

    return {**state, "extracted_fields": extracted, "confidence_scores": confidence, "extraction_method_log": method_log}
