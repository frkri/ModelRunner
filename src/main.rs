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
use crate::error::HttpErrorResponse;
use crate::error::ModelRunnerError;
use crate::model::task::instruct::{InstructHandler, InstructRequest, InstructResponse};

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
    static ref PHI2_MODEL: model::phi2::Phi2Model = model::phi2::Phi2Model {
        name: "Phi2".to_string(),
        download_url: "https://example.com/phi2".to_string(),
    };
    static ref PHI3_MODEL: model::phi3::Phi3Model = model::phi3::Phi3Model {
        name: "Phi3".to_string(),
        download_url: "https://example.com/phi3".to_string(),
    };
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

    let router = Router::new().route("/instruct", get(handle_instruct_request));

    let listener = TcpListener::bind(format!("{}:{}", config.address, config.port)).await?;
    info!("Listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => info!("Shutting down..."),
        Err(e) => error!("Failed to listen for shutdown signal: {}", e),
    }
}

#[axum_macros::debug_handler]
async fn handle_instruct_request(
    Json(req): Json<InstructRequest>,
) -> Result<(StatusCode, Json<InstructResponse>), ModelRunnerError> {
    match req.model.as_str() {
        "phi2" => Ok((StatusCode::OK, Json(PHI2_MODEL.run(req)?))),
        "phi3" => Ok((StatusCode::OK, Json(PHI3_MODEL.run(req)?))),
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}
