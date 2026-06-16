import datetime
from decimal import Decimal
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.errors import ExtractionError


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None


class Totals(BaseModel):
    """Monetary summary of the invoice."""

    currency: Optional[str] = None
    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    tax_rate: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    total: Optional[Decimal] = None
    amount_due: Optional[Decimal] = None


class InvoiceResponse(BaseModel):
    """Fully typed response for a successfully parsed invoice."""

    document_type: Literal["invoice"] = "invoice"
    issuer: Optional[str] = None
    issuer_address: Optional[str] = None
    recipient: Optional[str] = None
    date: Optional[datetime.date] = None
    due_date: Optional[datetime.date] = None
    totals: Totals = Field(default_factory=Totals)
    line_items: List[LineItem] = Field(default_factory=list)
    reference_ids: List[str] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence scores between 0.0 and 1.0",
    )
    extraction_method: str = Field(
        description="Which pipeline branch produced this result, e.g. 'docling+rule_based'"
    )
    requires_review: bool = Field(
        default=False,
        description="True when one or more key fields have low confidence and need human verification.",
    )
    review_reasons: List[str] = Field(
        default_factory=list,
        description="Explains why this document was flagged for human review.",
    )
    processing_time_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Time taken by each pipeline node in milliseconds.",
    )
    total_time_ms: float = Field(
        default=0.0,
        description="Total end-to-end processing time in milliseconds.",
    )
    errors: List[ExtractionError] = Field(
        default_factory=list,
        description="Field-level extraction failures. Never null — may be empty.",
    )
