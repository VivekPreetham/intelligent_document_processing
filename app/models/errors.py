from typing import Optional
from pydantic import BaseModel


class ExtractionError(BaseModel):
    """Represents a single field-level extraction failure."""

    field_name: str
    reason: str
    raw_value: Optional[str] = None
