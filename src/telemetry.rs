use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Compression, TonicExporterBuilder, WithExportConfig};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Config;
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing_opentelemetry::{MetricsLayer, OpenTelemetryLayer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;

static METER: OnceLock<SdkMeterProvider> = OnceLock::new();

pub(crate) fn init_telemetry(endpoint: &Option<String>, compress: bool) {
    let service_resource = Resource::new(vec![
        KeyValue::new(SERVICE_NAME, env!("CARGO_PKG_NAME")),
        KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(build_tonic_exporter(endpoint, compress))
        .with_trace_config(Config::default().with_resource(service_resource.clone()))
        .install_batch(runtime::Tokio)
        .context("Failed to install tracer")
        .unwrap();

    let meter = opentelemetry_otlp::new_pipeline()
        .metrics(runtime::Tokio)
        .with_exporter(build_tonic_exporter(endpoint, compress))
        .with_resource(service_resource)
        .build()
        .context("Failed to install meter")
        .unwrap();
    METER.set(meter.clone()).unwrap();

    global::set_text_map_propagator(TraceContextPropagator::new());
    Registry::default()
        .with(EnvFilter::try_from_default_env().unwrap_or(EnvFilter::new("INFO")))
        .with(tracing_subscriber::fmt::layer())
        .with(OpenTelemetryLayer::new(tracer))
        .with(MetricsLayer::new(meter))
        .init();
}

fn build_tonic_exporter(endpoint: &Option<String>, compress: bool) -> TonicExporterBuilder {
    let mut exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_timeout(Duration::from_secs(15));

    if let Some(endpoint) = endpoint {
        exporter = exporter.with_endpoint(endpoint);
    }
    if compress {
        exporter = exporter.with_compression(Compression::Gzip);
    }

    exporter
}

pub(crate) fn shutdown_meter_provider() {
    if let Some(meter) = METER.get() {
        meter
            .shutdown()
            .context("Failed to shutdown meter provider")
            .unwrap();
    }
}
