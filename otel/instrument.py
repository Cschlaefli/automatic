import logging
from os import environ
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
    DEFAULT_ENDPOINT
)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_ENDPOINT
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from opentelemetry.trace import get_tracer_provider, set_tracer_provider
from fastapi import FastAPI

logger = logging.getLogger(__name__)

def add_tracer() -> TracerProvider:
    resource = Resource(attributes={
        SERVICE_NAME: "sdnext"
    })

    tracerProvider = TracerProvider(resource=resource)
    tracerProvider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )
    set_tracer_provider(tracerProvider)
    logger.info("OpenTelemetry tracer provider registered")
    logger.info("otel exporter endpoint: %s", environ.get(OTEL_EXPORTER_OTLP_ENDPOINT, DEFAULT_ENDPOINT))
    return tracerProvider

def instrument_api(app: FastAPI):

    FastAPIInstrumentor.instrument_app(app,
                                       tracer_provider=get_tracer_provider(),
                                       excluded_urls="sdapi/v1/status,health/.*")
    if app._is_instrumented_by_opentelemetry:
        logger.info("app is instrumented by opentelemetry")
