use std::time::Duration;

use anyhow::Context;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{TonicExporterBuilder, WithExportConfig};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Config;
use opentelemetry_sdk::{runtime, Resource};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing_chrome::ChromeLayerBuilder;
use tracing_opentelemetry::{MetricsLayer, OpenTelemetryLayer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[tracing::instrument(level = "info")]
pub(crate) fn init_telemetry(
    endpoint: &Option<String>,
    console: bool,
    tracing_chrome: bool,
) -> Vec<impl Drop> {
    let service_resource = Resource::new(vec![
        KeyValue::new(SERVICE_NAME, env!("CARGO_PKG_NAME")),
        KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    // Builds the initial layer
    let mut guards = vec![];
    let mut layer = EnvFilter::try_from_default_env()
        .unwrap_or(EnvFilter::new("INFO"))
        .boxed();

    // Additions to the layer
    if let Some(endpoint) = endpoint {
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

        layer = layer
            .and_then(OpenTelemetryLayer::new(tracer))
            .and_then(MetricsLayer::new(meter))
            .boxed();
    }
    if endpoint.is_none() || console {
        layer = layer.and_then(tracing_subscriber::fmt::layer()).boxed();
    }
    if tracing_chrome {
        let (chrome_layer, chrome_guard) = ChromeLayerBuilder::new().build();
        guards.push(chrome_guard);

        layer = layer.and_then(chrome_layer).boxed();
    }

    global::set_text_map_propagator(TraceContextPropagator::new());
    tracing_subscriber::registry().with(layer).init();

    guards
}

#[tracing::instrument(level = "trace", skip(endpoint))]
fn build_tonic_exporter(endpoint: &String) -> TonicExporterBuilder {
    opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(15))
}
