use std::time::Duration;

use anyhow::Context;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{TonicExporterBuilder, WithExportConfig};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Config;
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing_opentelemetry::{MetricsLayer, OpenTelemetryLayer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;

pub(crate) fn init_telemetry(endpoint: &Option<String>, console: bool) {
    let service_resource = Resource::new(vec![
        KeyValue::new(SERVICE_NAME, env!("CARGO_PKG_NAME")),
        KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(build_tonic_exporter(endpoint))
        .with_trace_config(Config::default().with_resource(service_resource.clone()))
        .install_batch(runtime::Tokio)
        .context("Failed to install tracer")
        .unwrap();

    let meter = opentelemetry_otlp::new_pipeline()
        .metrics(runtime::Tokio)
        .with_exporter(build_tonic_exporter(endpoint))
        .with_resource(service_resource)
        .build()
        .context("Failed to install meter")
        .unwrap();

    global::set_text_map_propagator(TraceContextPropagator::new());
    let registry = Registry::default()
        .with(EnvFilter::try_from_default_env().unwrap_or(EnvFilter::new("INFO")))
        .with(OpenTelemetryLayer::new(tracer))
        .with(MetricsLayer::new(meter));

    if endpoint.is_none() || console {
        registry.with(tracing_subscriber::fmt::layer()).init();
    } else {
        registry.init();
    }
}

fn build_tonic_exporter(endpoint: &Option<String>) -> TonicExporterBuilder {
    let mut exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_timeout(Duration::from_secs(15));

    if let Some(endpoint) = endpoint {
        exporter = exporter.with_endpoint(endpoint);
    }

    exporter
}
