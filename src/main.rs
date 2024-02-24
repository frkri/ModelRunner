use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Multipart};
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use candle_transformers::models::mixformer;
use clap::Parser;
use clap_serde_derive::ClapSerde;
use hf_hub::api::sync::Api;
use lazy_static::lazy_static;
use log::{error, info};

use crate::config::Config;
use crate::error::ModelRunnerError;
use crate::error::{HttpErrorResponse, ModelResult};
use crate::inference::models::model::AudioTask;
use crate::inference::models::model::ModelBase;
use crate::inference::models::model::ModelDomain;
use crate::inference::models::model::TextTask;
use crate::inference::models::phi2::Phi2Model;
use crate::inference::models::phi2::Phi2ModelConfig;
use crate::inference::models::whisper::WhisperModel;
use crate::inference::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::inference::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::inference::task::transcribe::{
    TranscribeHandler, TranscribeRequest, TranscribeResponse,
};

mod config;
mod error;
mod inference;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, env, default_value = "ModelRunner.toml")]
    config_file: String,

    /// Configuration options
    #[command(flatten)]
    pub opt_config: <Config as ClapSerde>::Opt,
}

lazy_static! {
    static ref PHI2_MODEL: Phi2Model = Phi2Model::new(
        Api::new().expect("Failed to create API"),
        ModelBase {
            name: "Candle Phi2".into(),
            license: "MIT".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct,]),
            repo_id: "lmz/candle-quantized-phi".into(),
            repo_revision: "main".into(),
        },
        "tokenizer-puffin-phi-v2.json".into(),
        "model-puffin-phi-v2-q80.gguf".into(),
        mixformer::Config::puffin_phi_v2(),
        Phi2ModelConfig::default(),
    )
    .context("Failed to create Phi2 model")
    .unwrap();
    static ref WHISPER_MODEL: WhisperModel = WhisperModel::new(
        Api::new().expect("Failed to create API"),
        ModelBase {
            name: "Candle Whisper".into(),
            license: "MIT".into(),
            domain: ModelDomain::Audio(AudioTask::Transcribe),
            repo_id: "lmz/candle-whisper".into(),
            repo_revision: "main".into(),
        },
        "config-tiny.json".into(),
        "tokenizer-tiny.json".into(),
        "model-tiny-q4k.gguf".into(),
        "melfilters.bytes".into(),
    )
    .context("Failed to create Whisper model")
    .unwrap();
    static ref MISTRAL7B_MODEL: Mistral7BModel = Mistral7BModel::new(
        Api::new().expect("Failed to create API"),
        ModelBase {
            name: "Candle Mistral 7B".into(),
            license: "Apache-2.0".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct,]),
            repo_id: "lmz/candle-mistral".into(),
            repo_revision: "main".into(),
        },
        "tokenizer-mistral-7b.json".into(),
        "tokenizer.json".into(),
        mixformer::Config::mistral_7b(),
        Mistral7BModelConfig::default(),
    );
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let args = Args::parse();
    let config = match Config::from_toml(&args.config_file) {
        Ok(conf) => conf.merge(args.opt_config),
        Err(err) => {
            if args.config_file == "ModelRunner.toml" {
                Config::default().merge(args.opt_config)
            } else {
                exit_err!(
                    1,
                    "Failed to read configuration file {} with error: {}",
                    args.config_file,
                    err
                );
            }
        }
    };

    let text_router = Router::new()
        .route("/raw", post(handle_raw_request))
        .route("/instruct", post(handle_instruct_request));

    let audio_router = Router::new()
        .route("/transcribe", post(handle_transcribe_request))
        // 10 MB limit
        .layer(DefaultBodyLimit::max(10_000_000));

    let router = Router::new()
        .nest("/text", text_router)
        .nest("/audio", audio_router);

    let addr = format!("{}:{}", config.address, config.port)
        .parse::<SocketAddr>()
        .context("Failed to create socket from address and port")?;
    info!("Listening on {}", addr);
    info!(
        "Supported features: avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let shutdown_handle = Handle::new();
    tokio::spawn(shutdown_handler(shutdown_handle.clone()));

    match (config.tls.certificate, config.tls.private_key) {
        (Some(certificate), Some(private_key)) => {
            let tls_config = RustlsConfig::from_pem_file(certificate, private_key)
                .await
                .context("Failed to create TLS configuration")?;
            info!("TLS support for HTTPS enabled");
            axum_server::bind_rustls(addr, tls_config)
                .handle(shutdown_handle)
                .serve(router.into_make_service())
                .await?;
        }
        (None, None) => {
            axum_server::bind(addr)
                .handle(shutdown_handle)
                .serve(router.into_make_service())
                .await?
        }
        _ => exit_err!(
            1,
            "Both certificate and private key must be provided to enable TLS support."
        ),
    };

    Ok(())
}

async fn shutdown_handler(handle: Handle) {
    match tokio::signal::ctrl_c().await {
        Ok(()) => {
            info!("Received shutdown signal");
            handle.graceful_shutdown(Some(Duration::from_secs(45)));
        }
        Err(e) => error!("Failed to listen for shutdown signal: {}", e),
    }
}

#[axum_macros::debug_handler]
async fn handle_raw_request(
    Json(req): Json<RawRequest>,
) -> ModelResult<(StatusCode, Json<RawResponse>)> {
    match req.model.as_str() {
        "phi2" => Ok((StatusCode::OK, Json(PHI2_MODEL.clone().run_raw(req)?))),
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}

#[axum_macros::debug_handler]
async fn handle_instruct_request(
    Json(req): Json<InstructRequest>,
) -> ModelResult<(StatusCode, Json<InstructResponse>)> {
    match req.model.as_str() {
        "phi2" => Ok((StatusCode::OK, Json(PHI2_MODEL.clone().run_instruct(req)?))),
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}

#[axum_macros::debug_handler]
async fn handle_transcribe_request(
    mut multipart: Multipart,
) -> ModelResult<(StatusCode, Json<TranscribeResponse>)> {
    let mut opt_request = None;
    let mut opt_file_bytes = None;

    while let Some(field) = multipart.next_field().await? {
        if let Some(name) = field.name() {
            match name {
                "request_content" => {
                    if field
                        .content_type()
                        .map_or(false, |content| content != "application/json")
                    {
                        bail_runner!(
                            StatusCode::BAD_REQUEST,
                            "Invalid mime type in content-type header for request_content field"
                        );
                    }
                    opt_request = Some(Json::<TranscribeRequest>::from_bytes(
                        &field.bytes().await?,
                    )?)
                }
                "audio_content" => {
                    if field
                        .content_type()
                        .map_or(false, |content| !VALID_WAV_MIME_TYPES.contains(&content))
                    {
                        bail_runner!(
                            StatusCode::BAD_REQUEST,
                            "Invalid mime type in content-type header for audio_content field"
                        );
                    }
                    opt_file_bytes = Some(field.bytes().await?);
                }
                _ => bail_runner!(StatusCode::BAD_REQUEST, "Unknown field {}", name),
            }
        }
    }

    if opt_request.is_none() || opt_file_bytes.is_none() {
        let missing_field = if opt_request.is_none() {
            "request_content"
        } else {
            "audio_content"
        };
        bail_runner!(
            StatusCode::BAD_REQUEST,
            "Missing field {} in multipart form",
            missing_field
        );
    }
    let file_bytes = opt_file_bytes.unwrap().to_vec().into_boxed_slice();
    let request = opt_request.as_ref().unwrap();

    match request.model.to_lowercase().as_str() {
        "whisper" => Ok((
            StatusCode::OK,
            Json(
                WHISPER_MODEL
                    .clone()
                    .run_transcribe(file_bytes, &request.language)?,
            ),
        )),
        _ => bail_runner!(
            StatusCode::NOT_FOUND,
            "Model {} not found",
            &opt_request.unwrap().model
        ),
    }
}

// As per https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Containers#wave_wav
static VALID_WAV_MIME_TYPES: [&str; 4] =
    ["audio/wave", "audio/wav", "audio/x-wav", "audio/x-pn-wav"];

#[macro_export]
macro_rules! exit_err {
    ($msg:expr) => {
        {
            error!($msg);
            std::process::exit(1);
        }
    };
    ($code:expr, $msg:expr) => {
        {
            error!($msg);
            std::process::exit($code);
        }
    };
    ($code:expr, $fmt:expr $(, $arg:expr)*) => {
        {
            error!($fmt $(, $arg)*);
            std::process::exit($code);
        }
    };
}
