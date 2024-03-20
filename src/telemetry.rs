use std::time::Duration;

use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Config;
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;

pub(crate) fn init_telemetry(endpoint: String) {
    let service_resource = Resource::new(vec![
        KeyValue::new(SERVICE_NAME, env!("CARGO_PKG_NAME")),
        KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_timeout(Duration::from_secs(8))
        .with_endpoint(endpoint);

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(Config::default().with_resource(service_resource))
        .install_batch(runtime::Tokio)
        .unwrap();

    global::set_text_map_propagator(TraceContextPropagator::new());
    Registry::default()
        .with(EnvFilter::try_from_default_env().unwrap_or(EnvFilter::new("INFO")))
        .with(tracing_subscriber::fmt::layer())
        .with(OpenTelemetryLayer::new(tracer))
        .init();
}
