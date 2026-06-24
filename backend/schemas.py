"""
backend/schemas.py
==================
Pydantic request/response schemas for FastAPI endpoints.

Keeping schemas separate from database models ensures:
- API contract stays stable even if the DB schema changes.
- Response models strip internal fields (like raw ObjectId) before
  sending to the frontend.
- OpenAPI docs are auto-generated with accurate types.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class OCRResultResponse(BaseModel):
    """Single OCR result returned by the API."""
    id: Optional[str] = Field(default=None, description="MongoDB document ID")
    filename: str
    text: str
    length: int
    confidence: float
    status: str
    timestamp: Optional[datetime] = None

    model_config = {"populate_by_name": True}


class HistoryResponse(BaseModel):
    """Paginated history of OCR results."""
    total: int
    results: List[OCRResultResponse]


class DeleteResponse(BaseModel):
    """Response after deleting history."""
    deleted: int
    message: str


class StatusResponse(BaseModel):
    """Health check / status response."""
    app: str
    version: str
    mongodb: str   # "connected" | "disconnected"
    mqtt: str      # "connected" | "disconnected"
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: Optional[str] = None
