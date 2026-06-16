from __future__ import annotations

import logging

from app.models.errors import ExtractionError
from app.models.response import ErrorResponse
from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)


def error_node(state: IDPState) -> IDPState:
    """
    LangGraph node — Error exit.

    Reached when is_fatal_error=True (both parsers failed) or
    document_type='unknown' after classification.

    Builds a typed ErrorResponse so the API always returns structured JSON —
    never a raw 500 or an untyped dict — even on total pipeline failure.

    Reads:   fatal_error_message, validation_errors, document_type
    Writes:  final_response
    """
    message = state.get("fatal_error_message") or (
        "Document could not be classified. "
        "It may not be an invoice or supported transport document."
    )
    prior_errors = state.get("validation_errors") or []

    logger.error("Pipeline error exit — %s", message)

    response = ErrorResponse(
        status="error",
        message=message,
        errors=prior_errors,
    )

    return {**state, "final_response": response}
