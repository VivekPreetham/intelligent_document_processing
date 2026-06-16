from __future__ import annotations

import logging
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from app.pipeline.state import IDPState
from app.pipeline.timing import node_timer

logger = logging.getLogger(__name__)

# Confidence thresholds
CONFIDENCE_RULE_MATCH = 1.0      # clean regex match
CONFIDENCE_HEURISTIC = 0.7       # pattern found but needs normalisation
CONFIDENCE_LLM_FALLBACK = 0.0    # not found by rules — needs LLM
LLM_CALL_THRESHOLD = 0.7         # fields below this score go to LLM


# ── Rule-based helpers ────────────────────────────────────────────────────────

def _extract_invoice_number(text: str) -> Tuple[Optional[str], float]:
    patterns = [
        r"invoice\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
        r"inv\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
        r"invoice\s*no\.?\s*:?\s*([A-Z0-9\-\/]+)",
        r"bill\s*no\.?\s*:?\s*([A-Z0-9\-\/]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), CONFIDENCE_RULE_MATCH
    return None, CONFIDENCE_LLM_FALLBACK


def _extract_date(text: str, label: str = "invoice date") -> Tuple[Optional[str], float]:
    """
    Look for a date near a label keyword. Returns ISO 8601 string or None.
    Handles both inline (Date: Mar 06 2012) and multiline layouts where
    the label and value are on separate lines.
    """
    # Normalise whitespace/newlines so multiline labels become inline
    flat = re.sub(r"\s+", " ", text)

    patterns = [
        # Label followed immediately by date value (inline or after normalisation)
        rf"(?:{label}|date)\s*:?\s*(\d{{1,2}}[\/\-\.]\d{{1,2}}[\/\-\.]\d{{2,4}})",
        rf"(?:{label}|date)\s*:?\s*([A-Za-z]{{3,9}}\s+\d{{1,2}},?\s+\d{{4}})",
        rf"(?:{label}|date)\s*:?\s*(\d{{1,2}}\s+[A-Za-z]{{3,9}}\s+\d{{4}})",
        # Standalone date patterns (fallback)
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})",
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
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


def _extract_date_labelled(text: str, label_pattern: str) -> Tuple[Optional[str], float]:
    """
    Extract a date only when preceded by a specific label.
    Used for due_date to avoid matching the invoice date again.
    """
    flat = re.sub(r"\s+", " ", text)
    patterns = [
        rf"{label_pattern}\s*:?\s*(\d{{1,2}}[\/\-\.]\d{{1,2}}[\/\-\.]\d{{2,4}})",
        rf"{label_pattern}\s*:?\s*([A-Za-z]{{3,9}}\s+\d{{1,2}},?\s+\d{{4}})",
        rf"{label_pattern}\s*:?\s*(\d{{4}}-\d{{2}}-\d{{2}})",
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


def _extract_issuer(text: str) -> Tuple[Optional[str], float]:
    """
    The issuer (vendor/seller) is the company that issued the invoice.
    It typically appears BEFORE the 'Bill To:' section.
    We extract text from the top of the document up to 'Bill To:',
    then pick the first meaningful line that looks like a company name.
    """
    # Slice text before 'Bill To' — everything after is recipient info
    bill_to_match = re.search(r"bill\s*to\s*:?", text, re.IGNORECASE)
    pre_bill = text[:bill_to_match.start()] if bill_to_match else text[:500]

    lines = [l.strip() for l in pre_bill.splitlines() if l.strip()]
    skip_patterns = [
        r"^[\d\/\-\.\s,\$€£#]+$",           # numbers, symbols only
        r"^https?://",                        # URLs
        r"^[\w\s]{1,20}:\s*$",              # field labels like "Date:"
    ]
    skip_words = {"invoice", "bill", "receipt", "tax", "proforma", "commercial", "bill to", "ship to", "sold to"}
    # Docling renders image-only content (logos) as HTML/markdown artifacts
    skip_artifacts = {"<!-- image -->", "image", "logo"}

    for line in lines:
        if len(line) < 3:
            continue
        # Strip markdown heading markers before checking
        clean = re.sub(r"^[#*>\s]+", "", line).strip()
        if not clean or len(clean) < 3:
            continue
        if clean.lower() in skip_words:
            continue
        if clean.lower() in skip_artifacts:
            continue
        # Skip HTML comments and markdown image syntax
        if re.match(r"<!--.*-->|!\[.*\]", clean):
            continue
        if any(re.match(p, clean, re.IGNORECASE) for p in skip_patterns):
            continue
        return clean, CONFIDENCE_HEURISTIC

    return None, CONFIDENCE_LLM_FALLBACK


def _extract_total(text: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Returns (amount, currency, confidence).
    Handles both inline ('Total: $58.11') and multiline layouts where
    label and value appear on separate lines.
    """
    flat = re.sub(r"\s+", " ", text)
    patterns = [
        # Inline: "Total: $58.11" or "Balance Due: $58.11"
        r"(?:balance\s*due|total\s*amount|amount\s*due|grand\s*total|total)\s*:?\s*([\$€£¥]?)\s*([\d,]+\.?\d{0,2})",
        # Reverse: "$58.11 total"
        r"([\$€£¥])\s*([\d,]+\.?\d{0,2})\s*(?:total|due|balance)",
    ]
    _symbol_to_iso = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

    for pattern in patterns:
        m = re.search(pattern, flat, re.IGNORECASE)
        if m:
            symbol = m.group(1).strip()
            currency = _symbol_to_iso.get(symbol, symbol or None)
            amount_str = m.group(2).replace(",", "")
            try:
                Decimal(amount_str)
                return amount_str, currency, CONFIDENCE_RULE_MATCH
            except InvalidOperation:
                pass
    return None, None, CONFIDENCE_LLM_FALLBACK


def _extract_reference_ids(text: str) -> Tuple[List[str], float]:
    """Extract all invoice/PO/reference numbers from the document."""
    refs: List[str] = []
    patterns = [
        r"(?:invoice|inv)\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
        r"(?:purchase\s*order|p\.?o\.?)\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
        r"(?:reference|ref)\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
        r"(?:order)\s*#?\s*:?\s*([A-Z0-9\-\/]+)",
    ]
    # Common false-positive tokens to reject
    _skip = {"ID", "NO", "REF", "NUM", "NUMBER", "N", "A"}

    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = m.group(1).strip()
            if not val:
                continue
            if val.upper() in _skip:
                continue
            # Must contain at least one digit — partial words like "OICE" are false positives
            if not re.search(r"\d", val):
                continue
            if val not in refs:
                refs.append(val)
    confidence = CONFIDENCE_RULE_MATCH if refs else CONFIDENCE_LLM_FALLBACK
    return refs, confidence


def _extract_subtax(text: str) -> Tuple[Optional[str], Optional[str], float]:
    """Extract subtotal and shipping/tax amounts. Normalises whitespace first."""
    flat = re.sub(r"\s+", " ", text)
    subtotal, tax = None, None
    sub_m = re.search(
        r"(?:subtotal|sub\s*total)\s*:?\s*[\$€£¥]?\s*([\d,]+\.?\d{0,2})",
        flat, re.IGNORECASE
    )
    # Also catch "Shipping:" as a proxy for tax/fees when no explicit tax line
    tax_m = re.search(
        r"(?:tax|vat|gst|hst|shipping)\s*:?\s*[\$€£¥]?\s*([\d,]+\.?\d{0,2})",
        flat, re.IGNORECASE
    )
    if sub_m:
        subtotal = sub_m.group(1).replace(",", "")
    if tax_m:
        tax = tax_m.group(1).replace(",", "")
    confidence = CONFIDENCE_RULE_MATCH if (subtotal or tax) else CONFIDENCE_LLM_FALLBACK
    return subtotal, tax, confidence


# ── Line item extraction ──────────────────────────────────────────────────────

def _extract_line_items_rule(text: str) -> List[Dict[str, Any]]:
    """
    Extract line items from invoice text using pattern matching.
    Looks for rows that contain: description + quantity + unit price + amount.
    Handles both markdown table format (from docling) and plain text rows.
    """
    items: List[Dict[str, Any]] = []

    # Strategy 1: Markdown table rows (docling produces these)
    # Format: | Description | Qty | Unit Price | Amount |
    table_row = re.compile(
        r"\|\s*(.+?)\s*\|\s*([\d,]+\.?\d*)\s*\|\s*[\$€£¥]?\s*([\d,]+\.?\d*)\s*\|\s*[\$€£¥]?\s*([\d,]+\.?\d*)\s*\|"
    )
    for m in table_row.finditer(text):
        desc = m.group(1).strip()
        # Skip header rows
        if re.search(r"desc|product|item|qty|quantity|price|amount|total", desc, re.IGNORECASE):
            continue
        if re.search(r"^[-\s|]+$", desc):
            continue
        try:
            items.append({
                "description": desc,
                "quantity": m.group(2).replace(",", ""),
                "unit_price": m.group(3).replace(",", ""),
                "amount": m.group(4).replace(",", ""),
            })
        except Exception:
            continue

    if items:
        return items

    # Strategy 2: Plain text rows — "Description   Qty   Price   Amount"
    # Look for lines with 2+ numeric tokens that could be qty/price/amount
    line_pattern = re.compile(
        r"^(.+?)\s{2,}(\d+(?:\.\d+)?)\s{2,}[\$€£¥]?\s*(\d+(?:[,\.]\d+)?)\s{2,}[\$€£¥]?\s*(\d+(?:[,\.]\d+)?)$",
        re.MULTILINE,
    )
    for m in line_pattern.finditer(text):
        desc = m.group(1).strip()
        if re.search(r"subtotal|total|tax|shipping|discount|amount due", desc, re.IGNORECASE):
            continue
        if len(desc) < 2:
            continue
        try:
            items.append({
                "description": desc,
                "quantity": m.group(2).replace(",", ""),
                "unit_price": m.group(3).replace(",", ""),
                "amount": m.group(4).replace(",", ""),
            })
        except Exception:
            continue

    return items


def _llm_extract_line_items(text: str) -> List[Dict[str, Any]]:
    """
    Ask Groq to extract line items as a JSON array when rule-based fails.
    Each item: {description, quantity, unit_price, amount}.
    """
    import json
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
            "You are an invoice data extraction assistant. "
            "Extract all line items from the invoice text. "
            "Return a JSON object with a single key 'line_items' containing an array. "
            "Each item must have: description (string), quantity (number or null), "
            "unit_price (number or null), amount (number or null). "
            "Do NOT include subtotal, tax, shipping, or total rows — only product/service lines. "
            "If no line items exist, return {\"line_items\": []}."
        )

        human = f"Invoice text:\n\n{text[:3000]}\n\nExtract all line items."

        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )

        result = json.loads(response.content.strip())
        items = result.get("line_items", [])
        logger.info("LLM extracted %d line items", len(items))
        return items if isinstance(items, list) else []

    except Exception as exc:
        logger.error("LLM line item extraction failed: %s", exc, exc_info=True)
        return []


# ── LLM fallback ──────────────────────────────────────────────────────────────

def _llm_extract_invoice(text: str, missing_fields: List[str]) -> Dict[str, Any]:
    """
    Ask Groq to extract only the fields that rule-based extraction missed.
    Uses response_format=json_object to guarantee JSON output — no markdown
    fences or prose wrapping to strip.
    """
    import json

    # Only attempt fields we know about
    known_fields = {
        "invoice_number", "issuer", "date", "due_date",
        "total", "subtotal", "tax", "currency", "recipient",
    }
    requested = [f for f in missing_fields if f in known_fields]
    if not requested:
        logger.info("LLM skip — no known fields in missing list: %s", missing_fields)
        return {}

    fields_desc = ", ".join(requested)
    logger.info("LLM fallback — requesting fields: %s", fields_desc)

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        api_key = os.getenv("GROQ_API_KEY")
        logger.info("GROQ_API_KEY present: %s", bool(api_key))

        llm = ChatGroq(
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        system = (
            "You are an invoice data extraction assistant. "
            "You MUST respond with a single JSON object and nothing else. "
            f"Extract these fields from the invoice text: {fields_desc}. "
            "Rules:\n"
            "- dates → ISO 8601 (YYYY-MM-DD)\n"
            "- monetary amounts → plain numbers without currency symbols (e.g. 58.11 not $58.11)\n"
            "- if a field is not explicitly present in the text → null\n"
            "- due_date must only be set if the text contains an explicit 'Due Date', "
            "'Payment Due', or 'Pay By' label followed by a date. Do NOT infer it.\n"
            "- issuer is the company/person who ISSUED (sent) the invoice, "
            "not the recipient. Look for the vendor/seller name at the top of the document. "
            "NEVER return a value containing 'Bill To', 'Ship To', or 'Sold To' as the issuer — "
            "those are recipient sections. If you cannot find a clear vendor/seller name, return null.\n"
            "Never hallucinate values."
        )

        human = (
            f"Invoice text:\n\n{text[:3000]}\n\n"
            f"Return a JSON object with keys: {fields_desc}"
        )

        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )

        content = response.content.strip()
        logger.debug("LLM raw response: %s", content[:500])

        extracted = json.loads(content)
        logger.info("LLM extracted: %s", {k: v for k, v in extracted.items() if v is not None})
        return extracted

    except Exception as exc:
        logger.error("LLM invoice extraction failed: %s", exc, exc_info=True)
        return {}


# ── Main extraction node ──────────────────────────────────────────────────────

def extract_invoice_node(state: IDPState) -> IDPState:

    docling_text: str | None = state.get("docling_output")
    llamaparse_text: str | None = state.get("llamaparse_output")


    import time as _time
    _t0 = _time.perf_counter()

    parts = [t for t in [docling_text, llamaparse_text] if t]
    full_text = "\n\n---\n\n".join(parts)  # separator makes boundaries visible

    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    method_log: List[str] = []

    # ── Invoice number ────────────────────────────────────────────────────────
    inv_num, conf = _extract_invoice_number(full_text)
    extracted["invoice_number"] = inv_num
    confidence["invoice_number"] = conf
    method_log.append(f"invoice_number: {'rule_based' if conf > 0 else 'pending_llm'}")

    # ── Date ─────────────────────────────────────────────────────────────────
    date_val, conf = _extract_date(full_text, "invoice date")
    method_log.append(f"date: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")
    extracted["date"] = date_val
    confidence["date"] = conf

    # ── Due date — only extract if a specific due/payment date label exists ──
    due_date_val, conf = _extract_date_labelled(full_text, r"(?:due\s*date|payment\s*due|pay\s*by)")
    extracted["due_date"] = due_date_val
    confidence["due_date"] = conf
    method_log.append(f"due_date: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # ── Issuer ────────────────────────────────────────────────────────────────
    issuer, conf = _extract_issuer(full_text)
    extracted["issuer"] = issuer
    confidence["issuer"] = conf
    method_log.append(f"issuer: {'heuristic' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # ── Totals ────────────────────────────────────────────────────────────────
    total_amt, currency, conf = _extract_total(full_text)
    extracted["total"] = total_amt
    extracted["currency"] = currency
    confidence["total"] = conf
    confidence["currency"] = conf
    method_log.append(f"total: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    subtotal, tax, conf = _extract_subtax(full_text)
    extracted["subtotal"] = subtotal
    extracted["tax"] = tax
    confidence["subtotal"] = conf
    confidence["tax"] = conf
    method_log.append(f"subtotal/tax: {'rule_based' if conf >= CONFIDENCE_HEURISTIC else 'pending_llm'}")

    # ── Recipient (always sent to LLM — no reliable rule-based pattern) ──────
    extracted["recipient"] = None
    confidence["recipient"] = CONFIDENCE_LLM_FALLBACK
    method_log.append("recipient: pending_llm")

    # ── Reference IDs ─────────────────────────────────────────────────────────
    ref_ids, conf = _extract_reference_ids(full_text)
    extracted["reference_ids"] = ref_ids
    confidence["reference_ids"] = conf
    method_log.append(f"reference_ids: {'rule_based' if conf > 0 else 'pending_llm'}")

    # ── Line items ────────────────────────────────────────────────────────────
    line_items = _extract_line_items_rule(full_text)
    if not line_items:
        logger.info("Rule-based line item extraction found nothing — trying LLM")
        line_items = _llm_extract_line_items(full_text)
    extracted["line_items"] = line_items
    confidence["line_items"] = CONFIDENCE_RULE_MATCH if line_items else CONFIDENCE_LLM_FALLBACK
    method_log.append(f"line_items: {'rule_based' if line_items else 'not_found'} ({len(line_items)} items)")
    logger.info("Line items extracted: %d", len(line_items))

    # ── LLM fallback for low-confidence fields ────────────────────────────────
    missing_fields = [
        field for field, score in confidence.items()
        if score < LLM_CALL_THRESHOLD
    ]

    if missing_fields:
        logger.info("Calling LLM for missing fields: %s", missing_fields)
        llm_results = _llm_extract_invoice(full_text, missing_fields)

        # Sanitise LLM output: treat string "null"/"none"/"n/a" as None
        _null_strings = {"null", "none", "n/a", "na", "undefined", ""}
        for field, value in llm_results.items():
            if isinstance(value, str) and value.lower().strip() in _null_strings:
                value = None
            if value is not None and (extracted.get(field) is None):
                extracted[field] = value
                confidence[field] = 0.85  # LLM result — high but not certain
                # Update method log entry for this field
                method_log = [
                    log.replace(f"{field}: pending_llm", f"{field}: llm_fallback")
                    for log in method_log
                ]

    # Finalise method log — replace any remaining pending_llm with not_found
    method_log = [
        log.replace("pending_llm", "not_found") for log in method_log
    ]

    # Summarise the overall extraction method for the response field
    used_llm = any("llm_fallback" in m for m in method_log)
    used_docling = bool(docling_text)
    used_llamaparse = bool(llamaparse_text)

    source = "+".join(filter(None, [
        "docling" if used_docling else None,
        "llamaparse" if used_llamaparse else None,
    ]))
    method_summary = f"{source}+{'rule_based+llm_fallback' if used_llm else 'rule_based'}"

    extracted["extraction_method"] = method_summary

    # Record extraction node timing
    timings: dict = state.get("node_timings") or {}
    timings["extract"] = round((_time.perf_counter() - _t0) * 1000, 2)

    return {
        **state,
        "extracted_fields": extracted,
        "confidence_scores": confidence,
        "extraction_method_log": method_log,
        "node_timings": timings,
    }
