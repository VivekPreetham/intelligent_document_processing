"""
Evaluation script — field-level precision and recall across test PDFs.

Usage:
    1. Add 8-10 invoice PDFs to tests/test_pdfs/
    2. Fill in GROUND_TRUTH below with the expected values for each file
    3. Run: uv run python tests/evaluate.py

Outputs a table of per-field precision/recall and one failure case analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

BASE_URL = "http://localhost:8000"

# ── Ground truth ──────────────────────────────────────────────────────────────
# Fill in expected values for each PDF in your test set.
# Use None for fields you haven't manually verified.
# Keys must match filenames in tests/test_pdfs/.
GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    # "invoice_001.pdf": {
    #     "issuer": "Acme Corp",
    #     "date": "2024-01-15",
    #     "total": "1500.00",
    #     "invoice_number": "INV-2024-001",
    # },
}

FIELDS_TO_EVALUATE = ["issuer", "date", "total", "invoice_number", "reference_ids"]


def normalise(value: Any) -> Optional[str]:
    """Normalise a field value for comparison — lowercase, strip whitespace."""
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(sorted(str(v).lower().strip() for v in value))
    return str(value).lower().strip()


def extract_pdf(pdf_path: Path) -> Dict[str, Any]:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{BASE_URL}/extract",
            files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
        )
    return response.json()


def flatten_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested fields for evaluation."""
    flat = dict(result)
    if "totals" in result and isinstance(result["totals"], dict):
        flat["total"] = result["totals"].get("total")
    if "reference_ids" in result and result["reference_ids"]:
        flat["invoice_number"] = result["reference_ids"][0] if result["reference_ids"] else None
    return flat


def evaluate():
    test_pdfs_dir = Path(__file__).parent / "test_pdfs"
    pdf_files = sorted(test_pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in tests/test_pdfs/. Add some and re-run.")
        return

    print(f"Found {len(pdf_files)} PDFs to evaluate.\n")

    # Per-field counters
    field_tp: Dict[str, int] = {f: 0 for f in FIELDS_TO_EVALUATE}
    field_fp: Dict[str, int] = {f: 0 for f in FIELDS_TO_EVALUATE}
    field_fn: Dict[str, int] = {f: 0 for f in FIELDS_TO_EVALUATE}

    failure_cases = []
    results_log = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name} ...", end=" ", flush=True)
        try:
            result = extract_pdf(pdf_path)
            flat = flatten_response(result)
            gt = GROUND_TRUTH.get(pdf_path.name, {})

            file_result = {
                "file": pdf_path.name,
                "document_type": result.get("document_type"),
                "extraction_method": result.get("extraction_method"),
                "errors": result.get("errors", []),
                "fields": {},
            }

            for field in FIELDS_TO_EVALUATE:
                predicted = normalise(flat.get(field))
                expected = normalise(gt.get(field)) if gt else None

                file_result["fields"][field] = {
                    "predicted": predicted,
                    "expected": expected,
                }

                if expected is None:
                    # No ground truth — skip scoring but log the prediction
                    continue

                if predicted is not None and predicted == expected:
                    field_tp[field] += 1
                elif predicted is not None and predicted != expected:
                    field_fp[field] += 1
                    failure_cases.append({
                        "file": pdf_path.name,
                        "field": field,
                        "expected": expected,
                        "predicted": predicted,
                        "errors": result.get("errors", []),
                        "extraction_method": result.get("extraction_method"),
                    })
                else:
                    # predicted is None but expected is not
                    field_fn[field] += 1

            results_log.append(file_result)
            print("✓")

        except Exception as exc:
            print(f"✗ ERROR: {exc}")
            failure_cases.append({
                "file": pdf_path.name,
                "field": "all",
                "expected": None,
                "predicted": None,
                "errors": [str(exc)],
            })

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("=" * 60)

    for field in FIELDS_TO_EVALUATE:
        tp = field_tp[field]
        fp = field_fp[field]
        fn = field_fn[field]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{field:<20} {precision:>10.2%} {recall:>10.2%} {tp:>5} {fp:>5} {fn:>5}")

    print("=" * 60)

    # ── Print all predictions (for report) ───────────────────────────────────
    print("\n── Per-file extraction results ──")
    for r in results_log:
        print(f"\n{r['file']} [{r['document_type']}] via {r['extraction_method']}")
        if r["errors"]:
            print(f"  Errors: {r['errors']}")
        for field, vals in r["fields"].items():
            p = vals["predicted"] or "(not found)"
            e = vals["expected"] or "(no ground truth)"
            match = "✓" if vals["predicted"] == vals["expected"] and vals["expected"] else "~"
            print(f"  {match} {field}: {p} | expected: {e}")

    # ── Failure case analysis ─────────────────────────────────────────────────
    if failure_cases:
        print("\n── Failure case analysis ──")
        case = failure_cases[0]
        print(f"File:              {case['file']}")
        print(f"Field:             {case['field']}")
        print(f"Expected:          {case['expected']}")
        print(f"Predicted:         {case['predicted']}")
        print(f"Extraction method: {case.get('extraction_method', 'unknown')}")
        print(f"Pipeline errors:   {json.dumps(case.get('errors', []), indent=2)}")

    # Save full results to JSON for the report
    out_path = Path(__file__).parent / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"results": results_log, "failures": failure_cases},
            f, indent=2, default=str,
        )
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    evaluate()
