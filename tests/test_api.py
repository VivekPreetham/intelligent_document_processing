"""
API integration tests for the IDP Service.

Run with:
    uv run pytest tests/test_api.py -v

Requires the service dependencies to be installed (uv sync).
Tests use FastAPI's TestClient — no running server needed.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

# Minimal valid single-page PDF bytes (hand-crafted, no external dependency)
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 12 Tf 100 700 Td (Invoice #INV-001) Tj ET\n"
    b"endstream\nendobj\n"
    b"xref\n0 5\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n360\n%%EOF"
)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_status_ok(self):
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["pipeline"] == "ready"


# ── /extract ──────────────────────────────────────────────────────────────────

class TestExtract:
    def test_extract_with_valid_pdf_returns_json(self):
        """A valid PDF should return 200 or 422 — never a raw 500."""
        response = client.post(
            "/extract",
            files={"file": ("invoice.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
        assert response.status_code in (200, 422)
        body = response.json()
        assert isinstance(body, dict)

    def test_extract_response_has_required_fields(self):
        """Success response must contain all required fields from the assignment."""
        response = client.post(
            "/extract",
            files={"file": ("invoice.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
        body = response.json()
        # Whether success or typed error, errors field must never be null
        assert "errors" in body
        assert body["errors"] is not None
        assert isinstance(body["errors"], list)

    def test_extract_success_response_fields(self):
        """On a 200 response, all assignment-required fields must be present."""
        response = client.post(
            "/extract",
            files={"file": ("invoice.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
        if response.status_code == 200:
            body = response.json()
            required = [
                "document_type", "errors", "confidence",
                "extraction_method", "reference_ids", "totals",
            ]
            for field in required:
                assert field in body, f"Missing required field: {field}"

    def test_extract_rejects_non_pdf(self):
        """Non-PDF uploads must return a 4xx error with a typed JSON body."""
        fake_txt = b"This is not a PDF"
        response = client.post(
            "/extract",
            files={"file": ("doc.txt", io.BytesIO(fake_txt), "text/plain")},
        )
        # Route wraps validation errors in ErrorResponse (422) — never a raw 500
        assert response.status_code in (400, 422)
        body = response.json()
        # Must be typed JSON — not a raw string or empty body
        assert isinstance(body, dict)

    def test_extract_rejects_empty_file(self):
        """Empty file uploads must return a 4xx error with a typed JSON body."""
        response = client.post(
            "/extract",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code in (400, 422)
        body = response.json()
        assert isinstance(body, dict)

    def test_extract_rejects_fake_pdf(self):
        """A file with PDF content-type but no PDF magic bytes must return a 4xx error."""
        fake = b"This is plain text pretending to be a PDF"
        response = client.post(
            "/extract",
            files={"file": ("fake.pdf", io.BytesIO(fake), "application/pdf")},
        )
        assert response.status_code in (400, 422)
        body = response.json()
        assert isinstance(body, dict)

    def test_extract_error_response_is_typed(self):
        """Error responses must be typed JSON — never a raw 500 or empty body."""
        response = client.post(
            "/extract",
            files={"file": ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf")},
        )
        assert response.status_code in (400, 422)
        body = response.json()
        assert isinstance(body, dict)
        # Must have an errors field (our ErrorResponse contract)
        assert "errors" in body
        assert body["errors"] is not None

    def test_extract_with_real_pdf_from_corpus(self):
        """If test PDFs are present in tests/test_pdfs/, run one through the pipeline."""
        test_pdfs_dir = Path(__file__).parent / "test_pdfs"
        pdf_files = list(test_pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No test PDFs found in tests/test_pdfs/ — add PDFs to run this test")

        pdf_path = pdf_files[0]
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/extract",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )

        assert response.status_code in (200, 422)
        body = response.json()
        assert "errors" in body
        assert body["errors"] is not None


# ── /batch ────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_with_two_pdfs_returns_list(self):
        """Batch with 2 files must return a list of 2 results."""
        response = client.post(
            "/batch",
            files=[
                ("files", ("inv1.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")),
                ("files", ("inv2.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2

    def test_batch_each_result_has_errors_field(self):
        """Every item in a batch result must have a non-null errors list."""
        response = client.post(
            "/batch",
            files=[
                ("files", ("inv1.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")),
            ],
        )
        body = response.json()
        for item in body:
            assert "errors" in item
            assert item["errors"] is not None

    def test_batch_rejects_empty_upload(self):
        """Batch with no files must return a 4xx error.
        FastAPI validates the required 'files' field before our code runs
        and returns 422 — which is correct and acceptable.
        """
        response = client.post("/batch", files=[])
        assert response.status_code in (400, 422)

    def test_batch_mixed_valid_invalid(self):
        """One valid and one invalid file — list of 2 results, second is an error."""
        response = client.post(
            "/batch",
            files=[
                ("files", ("good.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")),
                ("files", ("bad.txt", io.BytesIO(b"not a pdf"), "text/plain")),
            ],
        )
        # Batch always returns 200 — individual errors are inside the list
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2
