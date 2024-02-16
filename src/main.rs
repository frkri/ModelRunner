use anyhow::Result;
use axum::extract::{DefaultBodyLimit, Multipart};
use axum::http::StatusCode;
use axum::routing::post;
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
use crate::models::model::{AudioTask, ModelBase, ModelDomain, TextTask};
use crate::models::phi2::{Phi2Model, Phi2ModelConfig};
use crate::models::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::models::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::models::task::transcribe::{TranscribeHandler, TranscribeRequest, TranscribeResponse};
use crate::models::whisper::WhisperModel;

mod config;
mod error;
mod models;

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
        "model-puffin-phi-v2-q80.gguf".into(),
        mixformer::Config::puffin_phi_v2(),
        Phi2ModelConfig::default(),
    )
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
    .unwrap();
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

    // TODO act on request cancellation
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
            error!($($msg)*);
            std::process::exit(1);
        }
    };
    ($code:expr, $msg:expr) => {
        {
            error!($($arg)*);
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
