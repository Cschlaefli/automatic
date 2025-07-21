import logging
from os import environ
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
    DEFAULT_ENDPOINT
)

from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
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

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.metrics import set_meter_provider
from prometheus_client import make_asgi_app

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def otel_setup():
    resource = Resource(attributes={
        SERVICE_NAME: "sdnext"
    })

    tracerProvider = TracerProvider(resource=resource)
    tracerProvider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )
    set_tracer_provider(tracerProvider)

    LoggingInstrumentor(
        resource=resource,
    ).instrument(set_logging_format=True)
    logger.info("OpenTelemetry logging instrumented")
    logger.info("OpenTelemetry tracer provider configured")
    logger.info("otel exporter endpoint: %s", environ.get(OTEL_EXPORTER_OTLP_ENDPOINT, DEFAULT_ENDPOINT))

    ThreadingInstrumentor(
        resource=resource,
    ).instrument()
    logger.info("OpenTelemetry threading instrumented")

    prometheus_reader = PrometheusMetricReader()
    set_meter_provider(MeterProvider(
        resource=resource,
        metric_readers=[prometheus_reader]
    ))
    logger.info("OpenTelemetry prometheus metric provider configured")

def instrument_api(app: FastAPI):
    logger.info("Instrumenting FastAPI application")
    FastAPIInstrumentor.instrument_app(app,
                                       excluded_urls="sdapi/v1/status,health/.*")
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Metrics endpoint mounted at /metrics")
    if app._is_instrumented_by_opentelemetry:
        logger.info("app is instrumented by opentelemetry")
