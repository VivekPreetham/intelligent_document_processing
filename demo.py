"""
Demo script — POST a PDF to /extract and pretty-print the JSON response.

Usage:
    uv run python demo.py path/to/invoice.pdf

Or with curl:
    curl -X POST http://localhost:8000/extract \
         -F "file=@path/to/invoice.pdf" | python -m json.tool
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python demo.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        # Fall back to urllib if httpx not installed
        import urllib.request
        import urllib.error

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        boundary = "----IDP_BOUNDARY"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            "http://localhost:8000/extract",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            result = json.loads(e.read())

        print(json.dumps(result, indent=2, default=str))
        return

    # httpx path (preferred)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"Uploading: {pdf_path.name} ({len(pdf_bytes):,} bytes)")
    print("Sending to http://localhost:8000/extract ...\n")

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "http://localhost:8000/extract",
            files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
        )

    print(f"HTTP {response.status_code}\n")
    print(json.dumps(response.json(), indent=2, default=str))


if __name__ == "__main__":
    main()
