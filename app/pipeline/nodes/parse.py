from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from app.models.errors import ExtractionError
from app.pipeline.state import IDPState

logger = logging.getLogger(__name__)


def _parse_with_docling(pdf_path: str) -> str | None:
    """
    Run docling's DocumentConverter on the PDF and return markdown text.
    Docling is layout-aware — it preserves table structure, column order,
    and heading hierarchy, making it superior for well-formatted invoices.
    Returns None on failure.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        logger.info("docling parsed %d chars", len(markdown))
        return markdown
    except Exception as exc:
        logger.warning("docling failed: %s", exc)
        return None


def _parse_with_llamaparse(pdf_path: str) -> str | None:
    """
    Send the PDF to LlamaParse and return LLM-optimised markdown.
    LlamaParse excels at scanned or low-quality PDFs where layout cues
    are degraded — it applies its own OCR and normalisation pass.
    Returns None on failure.
    """
    try:
        from llama_parse import LlamaParse

        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            logger.warning("LLAMA_CLOUD_API_KEY not set — skipping LlamaParse")
            return None

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
        )
        documents = parser.load_data(pdf_path)
        if not documents:
            logger.warning("LlamaParse returned no documents")
            return None

        markdown = "\n\n".join(doc.text for doc in documents)
        logger.info("LlamaParse parsed %d chars", len(markdown))
        return markdown
    except Exception as exc:
        logger.warning("LlamaParse failed: %s", exc)
        return None


def parse_node(state: IDPState) -> IDPState:
    """
    LangGraph node — Parse.

    Writes to state:
        docling_output      str | None
        llamaparse_output   str | None
        is_fatal_error      bool
        fatal_error_message str | None
        validation_errors   list[ExtractionError]

    Both parsers run independently. If at least one succeeds the graph
    continues to the classify node. If both fail, is_fatal_error is set
    True and the graph routes to the error exit node.
    """
    pdf_bytes: bytes = state["raw_pdf_bytes"]
    filename: str = state.get("filename", "document.pdf")

    # Write PDF bytes to a temporary file — both libraries need a file path.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docling_output = _parse_with_docling(tmp_path)
        llamaparse_output = _parse_with_llamaparse(tmp_path)
    finally:
        # Always clean up the temp file.
        Path(tmp_path).unlink(missing_ok=True)

    both_failed = docling_output is None and llamaparse_output is None

    if both_failed:
        logger.error("Both parsers failed for file: %s", filename)
        return {
            **state,
            "docling_output": None,
            "llamaparse_output": None,
            "is_fatal_error": True,
            "fatal_error_message": (
                "Both docling and LlamaParse failed to extract text from the document. "
                "The file may be corrupted, password-protected, or not a valid PDF."
            ),
            "validation_errors": [
                ExtractionError(
                    field_name="document",
                    reason="Both docling and LlamaParse returned no output",
                    raw_value=filename,
                )
            ],
        }

    logger.info(
        "Parse complete — docling: %s, LlamaParse: %s",
        "ok" if docling_output else "failed",
        "ok" if llamaparse_output else "failed",
    )

    return {
        **state,
        "docling_output": docling_output,
        "llamaparse_output": llamaparse_output,
        "is_fatal_error": False,
        "fatal_error_message": None,
        "validation_errors": [],
    }
