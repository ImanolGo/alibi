"""OpenTelemetry tracing to Arize Phoenix.

Import-safe: if the OpenTelemetry SDK is missing, everything degrades to a
no-op. Trace export is batched and non-blocking, so an unreachable Phoenix
never blocks a game.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_initialised = False


def init_tracing(endpoint: str | None = None) -> None:
    """Point OpenTelemetry at Phoenix. Safe to call more than once."""
    global _initialised
    if _initialised:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - opentelemetry is a default dep
        log.warning("opentelemetry not installed; tracing disabled")
        return

    from .config import get_settings

    endpoint = endpoint or get_settings().phoenix_endpoint
    provider = TracerProvider(resource=Resource.create({"service.name": "alibi"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _initialised = True


class NoopSpan:
    """Stand-in span for when tracing is unavailable."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_attributes(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        return None


def get_tracer() -> Any:
    from opentelemetry import trace

    return trace.get_tracer("alibi")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, ignoring any failure to set a given attribute."""
    try:
        tracer = get_tracer()
    except Exception:  # pragma: no cover - defensive
        yield NoopSpan()
        return
    with tracer.start_as_current_span(name) as current:
        with contextlib.suppress(Exception):
            for key, value in attributes.items():
                current.set_attribute(key, value)
        yield current
