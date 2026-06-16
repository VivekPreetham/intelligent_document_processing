"""
Timing utility for LangGraph pipeline nodes.

Usage inside any node:
    with node_timer(state, "parse"):
        ... do work ...
    return {**state, ...}

The context manager records elapsed milliseconds into
state["node_timings"]["parse"] automatically.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from app.pipeline.state import IDPState


@contextmanager
def node_timer(state: IDPState, node_name: str) -> Generator[None, None, None]:
    """
    Context manager that measures wall-clock time for a pipeline node
    and writes the result (in ms) into state["node_timings"].
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings: dict = state.get("node_timings") or {}
        timings[node_name] = round(elapsed_ms, 2)
        state["node_timings"] = timings
