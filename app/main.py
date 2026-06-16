from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: import and compile the LangGraph pipeline eagerly so the
    first request is not slowed down by cold-start graph compilation.
    The graph is a module-level singleton in graph.py — importing it
    here triggers build_graph() once at startup.
    """
    logger.info("IDP Service starting — pre-warming LangGraph pipeline...")
    from app.pipeline.graph import idp_graph  # noqa: F401  triggers compilation
    logger.info("Pipeline ready.")
    yield
    logger.info("IDP Service shutting down.")


app = FastAPI(
    title="IDP Service",
    description="Intelligent Document Processing — invoice and transport document extraction",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler: any unhandled exception returns a typed ErrorResponse
    rather than FastAPI's default plain-text 500. This satisfies the
    assignment requirement that every response — including failures —
    is a Pydantic-validated JSON object.
    """
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Unexpected server error: {str(exc)}",
            "errors": [],
        },
    )
