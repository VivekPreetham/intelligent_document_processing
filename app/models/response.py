from __future__ import annotations

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field

from app.models.errors import ExtractionError
from app.models.invoice import InvoiceResponse
from app.models.transport import AirwayBillResponse, SeawayBillResponse


# Discriminated union — FastAPI will serialize to the correct schema
# based on the document_type literal field.
DocumentResponse = Annotated[
    Union[InvoiceResponse, SeawayBillResponse, AirwayBillResponse],
    Field(discriminator="document_type"),
]


class ErrorResponse(BaseModel):
    """
    Returned when the pipeline fails entirely — e.g. the file is not a PDF,
    parsing crashed, or classification returned 'unknown'.
    Hallucinated or unparseable output is NEVER wrapped as success.
    """

    status: Literal["error"] = "error"
    message: str
    errors: List[ExtractionError] = Field(default_factory=list)
