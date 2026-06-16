import datetime
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.models.errors import ExtractionError
from app.models.invoice import InvoiceResponse, LineItem, Totals
from app.models.transport import AirwayBillResponse, SeawayBillResponse, TransportTotals
from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)


def _safe_decimal(value: Any) -> Optional[Decimal]:
    """Convert a raw extracted value to Decimal. Returns None on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def _build_invoice(fields: Dict[str, Any], confidence: Dict[str, float], method: str) -> InvoiceResponse:
    """
    Construct an InvoiceResponse from raw extracted fields.
    Each field is coerced to its target type individually — if one field
    fails, it becomes an ExtractionError in the errors list rather than
    rejecting the whole response. This is the 'partial success' principle.
    """
    errors: List[ExtractionError] = []

    # ── date ─────────────────────────────────────────────────────────────────
    date_val: Optional[datetime.date] = None
    raw_date = fields.get("date")
    if raw_date:
        try:
            from dateutil import parser as dateutil_parser
            parsed = dateutil_parser.parse(str(raw_date))
            date_val = datetime.date(parsed.year, parsed.month, parsed.day)
        except Exception:
            errors.append(ExtractionError(
                field_name="date",
                reason=f"Could not parse '{raw_date}' to a valid date",
                raw_value=str(raw_date),
            ))

    # ── due_date ──────────────────────────────────────────────────────────────
    due_date_val: Optional[datetime.date] = None
    raw_due = fields.get("due_date")
    if raw_due:
        try:
            from dateutil import parser as dateutil_parser
            parsed = dateutil_parser.parse(str(raw_due))
            due_date_val = datetime.date(parsed.year, parsed.month, parsed.day)
        except Exception:
            errors.append(ExtractionError(
                field_name="due_date",
                reason=f"Could not parse '{raw_due}' to a valid date",
                raw_value=str(raw_due),
            ))

    # ── totals ────────────────────────────────────────────────────────────────
    totals = Totals(
        currency=fields.get("currency"),
        subtotal=_safe_decimal(fields.get("subtotal")),
        tax=_safe_decimal(fields.get("tax")),
        total=_safe_decimal(fields.get("total")),
    )

    # Flag if total couldn't be parsed but a raw value existed
    if fields.get("total") and totals.total is None:
        errors.append(ExtractionError(
            field_name="totals.total",
            reason=f"Could not convert '{fields['total']}' to a valid decimal",
            raw_value=str(fields["total"]),
        ))

    # ── reference_ids ─────────────────────────────────────────────────────────
    ref_ids = fields.get("reference_ids", [])
    if not isinstance(ref_ids, list):
        ref_ids = [str(ref_ids)] if ref_ids else []

    # ── Build the model ───────────────────────────────────────────────────────
    try:
        response = InvoiceResponse(
            document_type="invoice",
            issuer=fields.get("issuer"),
            issuer_address=fields.get("issuer_address"),
            recipient=fields.get("recipient"),
            date=date_val,
            due_date=due_date_val,
            totals=totals,
            line_items=[],
            reference_ids=ref_ids,
            confidence=confidence,
            extraction_method=method,
            errors=errors,
        )
        return response
    except ValidationError as exc:
        # Catch any remaining Pydantic validation errors and surface them
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(ExtractionError(
                field_name=field,
                reason=error["msg"],
                raw_value=str(fields.get(field)),
            ))
        # Return a minimal valid response with the errors recorded
        return InvoiceResponse(
            extraction_method=method,
            confidence=confidence,
            errors=errors,
        )


def _build_seaway(fields: Dict[str, Any], confidence: Dict[str, float], method: str) -> SeawayBillResponse:
    errors: List[ExtractionError] = []

    date_val = None
    raw_date = fields.get("date_of_issue")
    if raw_date:
        try:
            from dateutil import parser as dateutil_parser
            _p = dateutil_parser.parse(str(raw_date))
            date_val = datetime.date(_p.year, _p.month, _p.day)
        except Exception:
            errors.append(ExtractionError(
                field_name="date_of_issue",
                reason=f"Could not parse '{raw_date}' to a valid date",
                raw_value=str(raw_date),
            ))

    totals = TransportTotals(
        freight_charges=_safe_decimal(fields.get("freight_charges")),
        total_charges=_safe_decimal(fields.get("total_charges")),
    )

    ref_ids = fields.get("reference_ids", [])
    if not isinstance(ref_ids, list):
        ref_ids = [str(ref_ids)] if ref_ids else []

    try:
        return SeawayBillResponse(
            bl_number=fields.get("bl_number"),
            shipper=fields.get("shipper"),
            consignee=fields.get("consignee"),
            notify_party=fields.get("notify_party"),
            vessel_name=fields.get("vessel_name"),
            voyage_number=fields.get("voyage_number"),
            port_of_loading=fields.get("port_of_loading"),
            port_of_discharge=fields.get("port_of_discharge"),
            date_of_issue=date_val,
            totals=totals,
            reference_ids=ref_ids,
            confidence=confidence,
            extraction_method=method,
            errors=errors,
        )
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(ExtractionError(field_name=field, reason=error["msg"]))
        return SeawayBillResponse(extraction_method=method, confidence=confidence, errors=errors)


def _build_airway(fields: Dict[str, Any], confidence: Dict[str, float], method: str) -> AirwayBillResponse:
    errors: List[ExtractionError] = []

    date_val = None
    raw_date = fields.get("date_of_issue")
    if raw_date:
        try:
            from dateutil import parser as dateutil_parser
            _p = dateutil_parser.parse(str(raw_date))
            date_val = datetime.date(_p.year, _p.month, _p.day)
        except Exception:
            errors.append(ExtractionError(
                field_name="date_of_issue",
                reason=f"Could not parse '{raw_date}' to a valid date",
                raw_value=str(raw_date),
            ))

    totals = TransportTotals(
        freight_charges=_safe_decimal(fields.get("freight_charges")),
        total_charges=_safe_decimal(fields.get("total_charges")),
    )

    ref_ids = fields.get("reference_ids", [])
    if not isinstance(ref_ids, list):
        ref_ids = [str(ref_ids)] if ref_ids else []

    try:
        return AirwayBillResponse(
            awb_number=fields.get("awb_number"),
            shipper=fields.get("shipper"),
            consignee=fields.get("consignee"),
            issuing_carrier=fields.get("issuing_carrier"),
            airport_of_departure=fields.get("airport_of_departure"),
            airport_of_destination=fields.get("airport_of_destination"),
            flight_number=fields.get("flight_number"),
            date_of_issue=date_val,
            totals=totals,
            reference_ids=ref_ids,
            confidence=confidence,
            extraction_method=method,
            errors=errors,
        )
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(ExtractionError(field_name=field, reason=error["msg"]))
        return AirwayBillResponse(extraction_method=method, confidence=confidence, errors=errors)


def validate_node(state: IDPState) -> IDPState:
    """
    LangGraph node — Validation.

    Reads:   document_type, extracted_fields, confidence_scores,
             extraction_method_log, validation_errors (from parse node)
    Writes:  final_response, validation_errors (merged)

    Behaviour:
    - Picks the correct Pydantic builder based on document_type.
    - Coerces each field individually — partial failures become ExtractionErrors,
      not a full response rejection.
    - Merges any errors accumulated earlier (e.g. from the parse node) into
      the final errors list.
    - Hallucinated or uncoercible values are never silently accepted.
    """
    doc_type = state.get("document_type", "unknown")
    fields = state.get("extracted_fields") or {}
    confidence = state.get("confidence_scores") or {}
    method_log = state.get("extraction_method_log") or []
    prior_errors = state.get("validation_errors") or []

    # Summarise extraction method from the log
    method = fields.get("extraction_method", "unknown")

    logger.info("Validating document_type=%s", doc_type)

    if doc_type == "invoice":
        response = _build_invoice(fields, confidence, method)
    elif doc_type == "seaway_bill":
        response = _build_seaway(fields, confidence, method)
    elif doc_type == "airway_bill":
        response = _build_airway(fields, confidence, method)
    else:
        # unknown / parse_failed — fatal error handled by error_node,
        # but as a safety net we return None here.
        logger.warning("validate_node reached with doc_type=%s — nothing to validate", doc_type)
        return {
            **state,
            "final_response": None,
            "validation_errors": prior_errors,
        }

    # Merge prior errors (e.g. parse warnings) into the model's error list
    if prior_errors:
        response.errors = prior_errors + response.errors

    logger.info(
        "Validation complete — %d field errors, method=%s",
        len(response.errors),
        method,
    )

    return {
        **state,
        "final_response": response,
        "validation_errors": response.errors,
    }
