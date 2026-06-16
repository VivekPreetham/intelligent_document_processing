import datetime
from decimal import Decimal
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.errors import ExtractionError


class TransportTotals(BaseModel):
    """Cargo weight / volume summary for transport documents."""

    currency: Optional[str] = None
    freight_charges: Optional[Decimal] = None
    total_charges: Optional[Decimal] = None
    gross_weight: Optional[str] = None
    measurement: Optional[str] = None


class SeawayBillResponse(BaseModel):
    """Typed response for a Sea Waybill / Bill of Lading."""

    document_type: Literal["seaway_bill"] = "seaway_bill"
    bl_number: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    notify_party: Optional[str] = None
    vessel_name: Optional[str] = None
    voyage_number: Optional[str] = None
    port_of_loading: Optional[str] = None
    port_of_discharge: Optional[str] = None
    place_of_delivery: Optional[str] = None
    date_of_issue: Optional[datetime.date] = None
    description_of_goods: Optional[str] = None
    totals: TransportTotals = Field(default_factory=TransportTotals)
    reference_ids: List[str] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(default_factory=dict)
    extraction_method: str
    requires_review: bool = Field(default=False)
    review_reasons: List[str] = Field(default_factory=list)
    processing_time_ms: Dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = Field(default=0.0)
    errors: List[ExtractionError] = Field(default_factory=list)


class AirwayBillResponse(BaseModel):
    """Typed response for an Air Waybill (AWB)."""

    document_type: Literal["airway_bill"] = "airway_bill"
    awb_number: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    issuing_carrier: Optional[str] = None
    airport_of_departure: Optional[str] = None
    airport_of_destination: Optional[str] = None
    flight_number: Optional[str] = None
    date_of_issue: Optional[datetime.date] = None
    description_of_goods: Optional[str] = None
    totals: TransportTotals = Field(default_factory=TransportTotals)
    reference_ids: List[str] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(default_factory=dict)
    extraction_method: str
    requires_review: bool = Field(default=False)
    review_reasons: List[str] = Field(default_factory=list)
    processing_time_ms: Dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = Field(default=0.0)
    errors: List[ExtractionError] = Field(default_factory=list)
