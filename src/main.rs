use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use candle_transformers::models::mixformer;
use clap::Parser;
use clap_serde_derive::ClapSerde;
use hf_hub::api::sync::Api;
use lazy_static::lazy_static;
use log::{error, info};
use tokio::net::TcpListener;

use crate::config::Config;
use crate::error::ModelRunnerError;
use crate::error::{HttpErrorResponse, ModelResult};
use crate::model::model::{ModelBase, ModelDomain, TextTask};
use crate::model::phi2::{Phi2Model, Phi2ModelConfig};
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
    static ref PHI2_MODEL: Mutex<Phi2Model> = Mutex::new(
        Phi2Model::new(
            Api::new().expect("Failed to create API"),
            ModelBase {
                name: "Candle Phi2".into(),
                license: "MIT".into(),
                domain: ModelDomain::Text(vec![
                    TextTask::Chat,
                    TextTask::Extract,
                    TextTask::Instruct,
                    TextTask::Sentiment,
                    TextTask::Translate,
                    TextTask::Identify,
                ]),
                repo_id: "lmz/candle-quantized-phi".into(),
                repo_revision: "main".into(),
            },
            "tokenizer-puffin-phi-v2.json".into(),
            "model-puffin-phi-v2-q4k.gguf".into(),
            mixformer::Config::puffin_phi_v2(),
            Phi2ModelConfig::default(),
        )
        .unwrap()
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
