# IDP Service — Intelligent Document Processing

An end-to-end FastAPI service for extracting structured data from invoice and transport documents (Sea Waybills, Air Waybills) using a **LangGraph pipeline** with dual parsing, hybrid rule-based + LLM extraction, and per-field confidence scoring.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Async, auto-docs via OpenAPI, Pydantic integration |
| Pipeline | LangGraph (StateGraph) | Explicit node routing, conditional edges, auditable state |
| Parser 1 | Docling | Layout-aware local parsing, markdown table output |
| Parser 2 | LlamaParse | Cloud OCR, handles scanned PDFs and image-heavy layouts |
| LLM | Groq — `llama-3.1-8b-instant` | Free tier, sub-second inference, JSON mode |
| Validation | Pydantic v2 | Typed models, discriminated union, Decimal for money |
| Package manager | uv | Fast, deterministic, pyproject.toml native |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐
│  parse_node │  docling (local) + LlamaParse (cloud) run independently
│             │  outputs merged → combined text for all downstream nodes
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  classify_node   │  keyword scoring → invoice / seaway_bill / airway_bill
│                  │  LLM fallback if rules score < 1 keyword hit
└──────┬───────────┘
       │
  ┌────┴─────────────────────────────┐
  │                                  │
  ▼                                  ▼
extract_invoice_node      extract_seaway/airway_node
  │                                  │
  │  1. Rule-based per field         │
  │  2. Batch LLM for low-confidence │
  │  3. Line item extraction         │
  └────────────┬─────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ validate_node │  Pydantic coercion, human-in-the-loop flagging,
       │               │  per-node timing, partial-success error list
       └───────┬───────┘
               │
               ▼
        Final JSON Response
```

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Groq API key — [console.groq.com](https://console.groq.com) (free)
- LlamaParse API key — [cloud.llamaindex.ai](https://cloud.llamaindex.ai) (free tier)

### Install

```bash
git clone https://github.com/VivekPreetham/intelligent_document_processing.git
cd intelligent_document_processing
uv sync
```

### Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

`.env` contents:

```
GROQ_API_KEY=your_groq_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Run

```bash
uv run uvicorn app.main:app --reload
```

Service starts at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

---

## API Endpoints

### `POST /extract` — Single document

Upload one PDF and receive a structured extraction result.

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.pdf;type=application/pdf"
```

**Response (invoice):**

```json
{
  "document_type": "invoice",
  "issuer": "SuperStore",
  "recipient": "Aaron Bergman",
  "date": "2012-03-06",
  "due_date": null,
  "totals": {
    "currency": "USD",
    "subtotal": "53.82",
    "tax": "4.29",
    "total": "58.11"
  },
  "line_items": [
    {
      "description": "Eldon Base for stackable storage shelf, platinum",
      "quantity": "1",
      "unit_price": "38.94",
      "amount": "38.94"
    }
  ],
  "reference_ids": ["36259"],
  "confidence": {
    "invoice_number": 1.0,
    "date": 1.0,
    "issuer": 0.85,
    "total": 1.0
  },
  "requires_review": false,
  "review_reasons": [],
  "processing_time_ms": {
    "parse": 8423.0,
    "classify": 2.3,
    "extract": 1247.0,
    "validate": 4.1
  },
  "total_time_ms": 9676.4,
  "extraction_method": "docling+llamaparse+rule_based+llm_fallback",
  "errors": []
}
```

---

### `POST /batch` — Multiple documents (concurrent)

Upload up to 20 PDFs. All processed concurrently via `asyncio.gather`.

```bash
# Windows (PowerShell)
curl.exe -X POST http://localhost:8000/batch `
  -F "files=@invoice1.pdf;type=application/pdf" `
  -F "files=@invoice2.pdf;type=application/pdf"

# macOS / Linux
curl -X POST http://localhost:8000/batch \
  -F "files=@invoice1.pdf;type=application/pdf" \
  -F "files=@invoice2.pdf;type=application/pdf"
```

Returns a JSON array — one result per file. A failure on one file does not affect others.

---

### `GET /health` — Liveness check

```bash
curl http://localhost:8000/health
# {"status": "ok", "pipeline": "ready"}
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

14 tests covering upload validation, bad inputs, batch limits, health check, and real PDF extraction (if test PDFs are placed in `tests/test_pdfs/`).

---

## Running the Demo Script

```bash
uv run python demo.py "path/to/invoice.pdf"
```

Prints a pretty-printed JSON response to the terminal.

---

## Running Evaluation (Precision / Recall)

Edit `GROUND_TRUTH` in `tests/evaluate.py` with known field values, then:

```bash
uv run python tests/evaluate.py
```

Outputs per-field precision and recall across all test PDFs.

---

## Document Types Supported

| Type | Detected by | Key fields extracted |
|---|---|---|
| Invoice | `invoice`, `bill to`, `amount due` keywords | issuer, recipient, date, due_date, totals, line_items, reference_ids |
| Sea Waybill / Bill of Lading | `bill of lading`, `shipper`, `port of loading` keywords | bl_number, shipper, consignee, ports, vessel, date_of_issue, freight |
| Air Waybill | `air waybill`, `awb`, `airport` keywords | awb_number, shipper, consignee, airports, flight, date_of_issue, charges |

---

## Key Design Decisions

### Dual Parser + Output Merging
Both docling and LlamaParse run in parallel (not sequentially). Their text outputs are concatenated with a separator before extraction. This means if docling misses a field (e.g. vendor name in a logo image), LlamaParse's OCR output may contain it, and vice versa.

### Hybrid Extraction (Rule-based → LLM)
Rule-based regex runs first for every field. Only fields with confidence below 0.7 are sent to the LLM — in a **single batched call**, not one call per field. This minimises LLM cost and latency while handling edge cases.

### Per-field Confidence Scores
Every extracted field carries a score: `1.0` = clean regex match, `0.85` = LLM extraction, `0.7` = heuristic, `0.0` = not found. Consumers can filter on confidence to decide how much to trust each value.

### Human-in-the-Loop Flagging
Documents where key fields (issuer, total, date, invoice number) have confidence below 0.7 are automatically flagged with `requires_review: true` and a list of `review_reasons`. This enables a human review queue without blocking the pipeline.

### Decimal not Float for Money
All monetary values use Python's `Decimal` type, not `float`. `float` cannot represent `0.1 + 0.2` exactly — a critical correctness issue for financial documents.

### Partial Success
A single bad field produces an `ExtractionError` in the errors list without rejecting the entire response. Downstream systems receive the best available data alongside a clear description of what failed.

### Async Pipeline
`asyncio.gather` in `/batch` processes files concurrently. LangGraph's blocking `invoke()` is offloaded to a thread pool via `run_in_executor` so it never blocks the FastAPI event loop.

### Per-node Timing
Every pipeline node records its wall-clock time. The response includes `processing_time_ms` per node and `total_time_ms`, making it easy to identify bottlenecks (typically: parse > extract > classify ≈ validate).

---

## Project Structure

```
idp-service/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, global exception handler
│   ├── api/
│   │   └── routes.py            # /extract, /batch, /health endpoints
│   ├── models/
│   │   ├── invoice.py           # InvoiceResponse, LineItem, Totals
│   │   ├── transport.py         # SeawayBillResponse, AirwayBillResponse
│   │   ├── errors.py            # ExtractionError
│   │   └── response.py          # DocumentResponse union, ErrorResponse
│   └── pipeline/
│       ├── state.py             # IDPState TypedDict
│       ├── graph.py             # LangGraph StateGraph assembly
│       ├── timing.py            # node_timer context manager
│       └── nodes/
│           ├── parse.py         # docling + LlamaParse
│           ├── classify.py      # keyword scoring + LLM fallback
│           ├── extract.py       # invoice field extraction
│           ├── extract_transport.py  # seaway/airway extraction
│           ├── validate.py      # Pydantic coercion, HITL flagging, timing
│           └── error_node.py    # fatal error exit
├── tests/
│   ├── test_api.py              # 14 pytest tests
│   ├── evaluate.py              # precision/recall evaluation
│   └── test_pdfs/               # place sample PDFs here
├── demo.py                      # CLI demo script
├── pyproject.toml
└── .env.example
```
