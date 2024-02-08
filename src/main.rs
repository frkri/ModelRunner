use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use clap_serde_derive::ClapSerde;
use lazy_static::lazy_static;
use log::{error, info};
use tokio::net::TcpListener;

use crate::config::Config;
use crate::error::ModelRunnerError;
use crate::error::{HttpErrorResponse, ModelResult};
use crate::model::task::raw::{RawHandler, RawRequest, RawResponse};

mod config;
mod error;
mod model;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, default_value = "ModelRunner.toml")]
    config_path: String,

    /// Configuration options
    #[command(flatten)]
    pub opt_config: <Config as ClapSerde>::Opt,
}

lazy_static! {
    static ref PHI2_MODEL: Mutex<model::phi2::Phi2Model> = Mutex::new(
        model::phi2::Phi2Model::new(
            "lmz/candle-quantized-phi".into(),
            "main".into(),
            "model-puffin-phi-v2-q4k.gguf".into(),
            Some(299792458),
            Some(0.8),
            Some(0.2),
        )
        .expect("Failed to initialize Phi2Model")
    );
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let args = Args::parse();
    let config = Config::from_toml(&args.config_path)
        .context(format!(
            "Failed to load config from file {}",
            &args.config_path
        ))?
        .merge(args.opt_config);

    // TODO finish routes
    // TODO act on request cancellation
    let router = Router::new().route("/raw", get(handle_raw_request));

    let listener = TcpListener::bind(format!("{}:{}", config.address, config.port)).await?;
    info!("Listening on {}", listener.local_addr().unwrap());
    info!(
        "Supported features: avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

// TODO set timeout for shutdown signal
async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => info!("Shutting down..."),
        Err(e) => error!("Failed to listen for shutdown signal: {}", e),
    }
}

#[axum_macros::debug_handler]
async fn handle_raw_request(
    Json(req): Json<RawRequest>,
) -> ModelResult<(StatusCode, Json<RawResponse>)> {
    match req.model.as_str() {
        "phi2" => {
            let mut model = PHI2_MODEL.lock().unwrap(); // TODO try remove mutex?
            Ok((StatusCode::OK, Json(model.run(req)?)))
        }
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}
