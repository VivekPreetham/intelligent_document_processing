from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Union

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.response import ErrorResponse
from app.pipeline.graph import idp_graph

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def _read_and_validate_upload(file: UploadFile) -> tuple[bytes, str]:
    """
    Read upload bytes and validate it looks like a PDF.
    Raises HTTPException with a clear message on failure so the caller
    gets a typed 400 — not a raw 500.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are accepted.",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB.",
        )

    # PDF magic bytes check — every valid PDF starts with %PDF-
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF (missing PDF header).",
        )

    return pdf_bytes, file.filename or "document.pdf"


def _invoke_pipeline(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Invoke the compiled LangGraph pipeline synchronously.
    Returns the final state dict.
    """
    initial_state = {
        "raw_pdf_bytes": pdf_bytes,
        "filename": filename,
        "docling_output": None,
        "llamaparse_output": None,
        "document_type": None,
        "extracted_fields": None,
        "confidence_scores": None,
        "extraction_method_log": None,
        "validation_errors": None,
        "final_response": None,
        "is_fatal_error": False,
        "fatal_error_message": None,
    }
    return idp_graph.invoke(initial_state)


async def _process_single(file: UploadFile) -> Dict[str, Any]:
    """
    Process one PDF upload end-to-end.
    Returns a serialisable dict of the final response (success or error).
    Always returns a response — never raises inside the batch loop.
    """
    try:
        pdf_bytes, filename = await _read_and_validate_upload(file)
        logger.info("Processing file: %s (%d bytes)", filename, len(pdf_bytes))

        # Run the blocking graph.invoke in a thread pool so it doesn't
        # block the async event loop — important for /batch concurrency.
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None, _invoke_pipeline, pdf_bytes, filename
        )

        response = final_state.get("final_response")

        if response is None:
            return ErrorResponse(
                status="error",
                message="Pipeline completed but produced no output.",
                errors=[],
            ).model_dump(mode="json")

        # mode="json" converts datetime.date → ISO string, Decimal → float, etc.
        return response.model_dump(mode="json")

    except HTTPException as exc:
        return ErrorResponse(
            status="error",
            message=exc.detail,
            errors=[],
        ).model_dump()

    except Exception as exc:
        logger.exception("Unhandled error processing file: %s", exc)
        return ErrorResponse(
            status="error",
            message=f"Internal pipeline error: {str(exc)}",
            errors=[],
        ).model_dump()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/extract",
    summary="Extract structured data from a single PDF",
    response_description="Pydantic-validated extraction result or typed error response",
)
async def extract(file: UploadFile = File(...)):
    """
    Upload a single PDF invoice or transport document.

    Returns a fully typed JSON response:
    - **InvoiceResponse** for invoices
    - **SeawayBillResponse** for sea waybills / bills of lading
    - **AirwayBillResponse** for air waybills
    - **ErrorResponse** if the pipeline fails (never a raw 500)
    """
    result = await _process_single(file)

    # Return 422 status for error responses so clients can distinguish
    # success from pipeline failures at the HTTP level.
    if result.get("status") == "error":
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(status_code=200, content=result)


@router.post(
    "/batch",
    summary="Extract structured data from multiple PDFs",
    response_description="List of extraction results, one per uploaded file",
)
async def batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF files and receive a list of extraction results.

    All files are processed **concurrently** via asyncio.gather.
    Each result in the list is independently either a success response
    or a typed ErrorResponse — a failure on one file does not affect others.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch request.")

    logger.info("Batch processing %d files", len(files))

    results = await asyncio.gather(*[_process_single(f) for f in files])
    return JSONResponse(status_code=200, content=list(results))


@router.get(
    "/health",
    summary="Liveness check",
    response_description="Service and pipeline readiness status",
)
async def health():
    """
    Confirms the FastAPI service is running and the LangGraph pipeline
    is compiled and ready to accept requests.
    """
    pipeline_ready = idp_graph is not None
    return JSONResponse(
        status_code=200 if pipeline_ready else 503,
        content={
            "status": "ok" if pipeline_ready else "degraded",
            "pipeline": "ready" if pipeline_ready else "not_ready",
        },
    )
