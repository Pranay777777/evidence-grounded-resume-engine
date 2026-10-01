"""Request IDs, a server span per request, and one structured log line (ADR-018)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry.trace import SpanKind

from grounded.observability import tracer

log = logging.getLogger("grounded.request")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id(request: Request) -> str:
    """The caller's X-Request-ID if it is safe to echo and log; otherwise a new one."""
    supplied = request.headers.get("x-request-id", "")
    return supplied if _SAFE_ID.fullmatch(supplied) else uuid.uuid4().hex


def observe(app: FastAPI) -> None:
    @app.middleware("http")
    async def _observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request_id(request)
        request.state.request_id = rid
        started = time.perf_counter()
        # INTERNAL, not SERVER: FastAPI 0.142+ emits its own server span, and a
        # request should have one. Ours carries the request ID either way.
        with tracer.start_as_current_span(f"HTTP {request.method}", kind=SpanKind.INTERNAL) as span:
            span.set_attributes(
                {
                    "grounded.request_id": rid,
                    "http.request.method": request.method,
                    "url.path": request.url.path,
                }
            )
            response = await call_next(request)
            route = request.scope.get("route")
            template = getattr(route, "path", request.url.path)
            span.update_name(f"HTTP {request.method} {template}")
            span.set_attributes(
                {"http.route": template, "http.response.status_code": response.status_code}
            )
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": rid,
                    "method": request.method,
                    "route": template,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    **getattr(request.state, "log_fields", {}),
                }
            },
        )
        return response
