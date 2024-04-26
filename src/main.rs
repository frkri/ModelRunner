#![warn(
    clippy::correctness,
    clippy::complexity,
    clippy::suspicious,
    clippy::pedantic,
    clippy::cargo,
    clippy::perf,
    clippy::style,
    clippy::nursery
)]
#![allow(
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::multiple_crate_versions,
    clippy::cargo_common_metadata
)]

use std::net::SocketAddr;
use std::option::Option;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::MatchedPath;
use axum::extract::{DefaultBodyLimit, FromRef, Multipart, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use axum::{middleware, Extension, Json, Router};
use axum_extra::headers::authorization::Bearer;
use axum_extra::headers::Authorization;
use axum_extra::TypedHeader;
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use candle_transformers::models::mixformer;
use clap::Parser;
use clap_serde_derive::ClapSerde;
use hf_hub::api::sync::Api;
use lazy_static::lazy_static;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tower_http::trace::TraceLayer;
use tracing::instrument;
use tracing::{error, info, warn};

#[cfg(unix)]
use tikv_jemallocator::Jemalloc;

use crate::api::auth::{Auth, AuthToken};
use crate::api::client::{ApiClient, ApiClientCreateRequest, ApiClientDeleteRequest, Permission};
use crate::api::client::{ApiClientStatusRequest, ApiClientUpdateRequest};
use crate::config::Config;
use crate::error::ModelRunnerError;
use crate::error::{HttpErrorResponse, ModelResult};
use crate::inference::model_config::GeneralModelConfig;
use crate::inference::models::mistral7b::Mistral7BModel;
use crate::inference::models::model::AudioTask;
use crate::inference::models::model::ModelBase;
use crate::inference::models::model::ModelDomain;
use crate::inference::models::model::TextTask;
use crate::inference::models::openhermes::OpenHermesModel;
use crate::inference::models::phi::PhiModel;
use crate::inference::models::stablelm2::StableLm2Model;
use crate::inference::models::whisper::WhisperModel;
use crate::inference::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::inference::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::inference::task::transcribe::{
    TranscribeHandler, TranscribeRequest, TranscribeResponse,
};
use crate::telemetry::init_telemetry;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub mod api;
mod config;
pub mod error;
mod inference;
mod telemetry;

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

#[derive(Debug, Clone, FromRef)]
struct AppState {
    db_pool: SqlitePool,
    auth: Auth,
}

lazy_static! {
    static ref PHI2_MODEL: PhiModel = PhiModel::new(
        &Api::new().expect("Failed to create API"),
        &ModelBase {
            name: "Quantized Puffin Phi2".into(),
            license: "MIT".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct]),
            repo_id: "lmz/candle-quantized-phi".into(),
            repo_revision: "main".into(),
        },
        "lmz/candle-quantized-phi",
        "tokenizer-puffin-phi-v2.json",
        "model-puffin-phi-v2-q80.gguf",
        Some(mixformer::Config::puffin_phi_v2()),
        GeneralModelConfig::default(),
        false,
    )
    .map_err(|e| error!("Failed to create Phi2 model: {}", e))
    .unwrap();
    static ref PHI3_MODEL: PhiModel = PhiModel::new(
        &Api::new().expect("Failed to create API"),
        &ModelBase {
            name: "Quantized Phi3 Instruct".into(),
            license: "MIT".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct]),
            repo_id: "microsoft/Phi-3-mini-4k-instruct-gguf".into(),
            repo_revision: "main".into(),
        },
        "microsoft/Phi-3-mini-4k-instruct",
        "tokenizer.json",
        "Phi-3-mini-4k-instruct-q4.gguf",
        None,
        GeneralModelConfig::default(),
        true,
    )
    .map_err(|e| error!("Failed to create Phi3 model: {}", e))
    .unwrap();
    static ref WHISPER_MODEL: WhisperModel = WhisperModel::new(
        Api::new().expect("Failed to create API"),
        &ModelBase {
            name: "Quantized Whisper".into(),
            license: "MIT".into(),
            domain: ModelDomain::Audio(AudioTask::Transcribe),
            repo_id: "lmz/candle-whisper".into(),
            repo_revision: "main".into(),
        },
        "config-tiny.json",
        "tokenizer-tiny.json",
        "model-tiny-q4k.gguf",
        "melfilters.bytes",
    )
    .map_err(|e| error!("Failed to create Whisper model: {}", e))
    .unwrap();
    static ref MISTRAL7B_INSTRUCT_MODEL: Mistral7BModel = Mistral7BModel::new(
        &Api::new().expect("Failed to create API"),
        ModelBase {
            name: "Quantized Mistral7B Instruct".into(),
            license: "Apache 2.0".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct,]),
            repo_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
            repo_revision: "main".into(),
        },
        "tokenizer.json",
        "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
        GeneralModelConfig::default(),
    )
    .map_err(|e| error!("Failed to create Mistral7B model: {}", e))
    .unwrap();
    static ref OPENHERMES_MODEL: OpenHermesModel = OpenHermesModel::new(
        &Api::new().expect("Failed to create API"),
        ModelBase {
            name: "Quantized OpenHermes-2.5 Mistral7B".into(),
            license: "Apache 2.0".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct,]),
            repo_id: "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF".into(),
            repo_revision: "main".into(),
        },
        "tokenizer.json",
        "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        GeneralModelConfig::default(),
    )
    .map_err(|e| error!("Failed to create OpenHermes model: {}", e))
    .unwrap();
    static ref STABLELM2_ZEPHYR_MODEL: StableLm2Model = StableLm2Model::new(
        &Api::new().expect("Failed to create API"),
        &ModelBase {
            name: "Quantized StableLM 2 Zephyr 1.6B".into(),
            license: "StabilityAI Non-Commercial Research Community License".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct]),
            repo_id: "lmz/candle-stablelm".into(),
            repo_revision: "main".into(),
        },
        "tokenizer-gpt4.json",
        "stablelm-2-zephyr-1_6b-q4k.gguf",
        &GeneralModelConfig::default(),
        true,
    )
    .map_err(|e| error!("Failed to create StableLM2 model: {}", e))
    .unwrap();
    static ref STABLELM2_MODEL: StableLm2Model = StableLm2Model::new(
        &Api::new().expect("Failed to create API"),
        &ModelBase {
            name: "Quantized StableLM 2 1.6B".into(),
            license: "StabilityAI Non-Commercial Research Community License".into(),
            domain: ModelDomain::Text(vec![TextTask::Chat, TextTask::Instruct]),
            repo_id: "lmz/candle-stablelm".into(),
            repo_revision: "main".into(),
        },
        "tokenizer-gpt4.json",
        "stablelm-2-1_6b-q4k.gguf",
        &GeneralModelConfig::default(),
        false,
    )
    .map_err(|e| error!("Failed to create StableLM2 model: {}", e))
    .unwrap();
}

#[allow(clippy::too_many_lines)]
#[tokio::main]
#[instrument]
async fn main() -> Result<()> {
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

    // Init telemetry
    let _guards = init_telemetry(&config.otel_endpoint, config.console, config.trace_local);

    info!(
        "model_runner v{}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown")
    );
    info!(
        "Supported features: avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let sqlite_options = SqliteConnectOptions::new()
        .create_if_missing(true)
        .filename(config.sqlite_file_path);
    let db_pool = SqlitePool::connect_with(sqlite_options)
        .await
        .context("Failed to connect to Sqlite")?;
    sqlx::migrate!()
        .run(&db_pool)
        .await
        .context("Failed to run migrations")?;
    let app_state = AppState {
        db_pool,
        auth: Auth::default(),
    };

    let text_router = Router::new()
        .route("/raw", post(handle_raw_request))
        .route("/instruct", post(handle_instruct_request));

    let audio_router = Router::new()
        .route("/transcribe", post(handle_transcribe_request))
        // 10 MB limit
        .layer(DefaultBodyLimit::max(10_000_000));

    let auth_router = Router::new()
        .route("/status", post(handle_status_request))
        .route("/create", post(handle_create_request))
        .route("/delete", post(handle_delete_request))
        .route("/update", post(handle_update_request));

    let router = Router::new()
        .nest("/auth", auth_router)
        .nest("/text", text_router)
        .nest("/audio", audio_router)
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth_middleware,
        ))
        .route("/health", get(handle_health_request))
        .layer(TraceLayer::new_for_http())
        .layer(middleware::from_fn(track_request))
        .with_state(app_state);

    let addr = format!("{}:{}", config.address, config.port)
        .parse::<SocketAddr>()
        .context("Failed to create socket from address and port")?;
    info!("Listening on {}", addr);

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
                .await?;
        }
        _ => exit_err!(
            1,
            "Both certificate and private key must be provided to enable TLS support."
        ),
    };

    Ok(())
}

#[allow(clippy::redundant_pub_crate)]
#[tracing::instrument(level = "info", skip(handle))]
async fn shutdown_handler(handle: Handle) {
    let ctrl_c_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to create ctrl-c signal");
    };

    #[cfg(unix)]
    let terminate_signal = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
                info!("Received terminate signal");
            }
            Err(e) => error!("Failed to listen for terminate signal: {}", e),
        }
    };

    #[cfg(not(unix))]
    let terminate_signal = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c_signal => handle.graceful_shutdown(Some(Duration::from_secs(45))),
        () = terminate_signal => handle.graceful_shutdown(Some(Duration::from_secs(45))),
    }
}

#[instrument(skip_all)]
async fn auth_middleware(
    State(state): State<AppState>,
    TypedHeader(auth_header): TypedHeader<Authorization<Bearer>>,
    mut request: Request,
    next: Next,
) -> ModelResult<Response> {
    let client = ApiClient::with_token(
        &state.auth,
        AuthToken::from_raw_str(auth_header.token())?,
        &state.db_pool,
    )
    .await
    .map_err(|_| runner!(StatusCode::UNAUTHORIZED, "Failed to authenticate client"))?;
    client.has_permission(&Permission::USE_SELF)?;

    request.extensions_mut().insert(client);

    info!(monotonic_counter.requests_authorized = 1);
    Ok(next.run(request).await)
}

#[tracing::instrument(level = "trace", skip(request))]
fn get_scheme(request: &Request) -> String {
    request
        .uri()
        .scheme_str()
        .map_or_else(|| "http".to_string(), ToString::to_string)
}

#[tracing::instrument(level = "trace", skip(request))]
fn get_path(request: &Request) -> String {
    request.extensions().get::<MatchedPath>().map_or_else(
        || request.uri().path().to_string(),
        |matched_path| matched_path.as_str().to_string(),
    )
}

#[instrument(skip_all)]
async fn track_request(req: Request, next: Next) -> ModelResult<Response> {
    let start = Instant::now();
    let method = req.method().to_owned();
    let path = get_path(&req);
    let version = req.version();
    let scheme = get_scheme(&req);

    info!(counter.http.server.active_requests = 1, ?method);
    let response = next.run(req).await;
    info!(counter.http.server.active_requests = -1, ?method);
    info!(
        histogram.http.server.request.duration = start.elapsed().as_secs_f64(),
        ?method,
        path,
        ?version,
        scheme
    );

    Ok(response)
}

#[tracing::instrument(level = "trace", skip())]
#[axum_macros::debug_handler]
async fn handle_health_request() -> ModelResult<StatusCode> {
    Ok(StatusCode::OK)
}

#[tracing::instrument(level = "trace", skip(req))]
#[axum_macros::debug_handler]
async fn handle_status_request(
    State(state): State<AppState>,
    Extension(mut client): Extension<ApiClient>,
    req: Option<Json<ApiClientStatusRequest>>,
) -> ModelResult<(StatusCode, Json<ApiClient>)> {
    if let Some(req) = req {
        client.has_permission(&Permission::STATUS_OTHER)?;
        client = ApiClient::with_id(req.id.as_str(), &state.db_pool)
            .await
            .map_err(|_| {
                runner!(
                    StatusCode::NOT_FOUND,
                    "Failed to find any client matching ID"
                )
            })?;
    } else {
        client.has_permission(&Permission::STATUS_SELF)?;
    }

    Ok((StatusCode::OK, Json(client)))
}

#[tracing::instrument(level = "trace", skip())]
#[axum_macros::debug_handler]
async fn handle_create_request(
    State(state): State<AppState>,
    Extension(client): Extension<ApiClient>,
    Json(req): Json<ApiClientCreateRequest>,
) -> ModelResult<(StatusCode, Json<ApiClient>)> {
    client.has_permission(&Permission::CREATE_SELF)?;
    let client = ApiClient::new(
        &state.auth,
        &req.name,
        &req.permissions.iter().cloned().collect::<Permission>(),
        &Some(client.token.id),
        &state.db_pool,
    )
    .await?;
    Ok((StatusCode::OK, Json(client)))
}

#[tracing::instrument(level = "trace", skip())]
#[axum_macros::debug_handler]
async fn handle_delete_request(
    State(state): State<AppState>,
    Extension(mut client): Extension<ApiClient>,
    Json(req): Json<ApiClientDeleteRequest>,
) -> ModelResult<StatusCode> {
    client.has_permission(&Permission::DELETE_SELF)?;
    if req.id != client.token.id {
        client = ApiClient::with_id(req.id.as_str(), &state.db_pool).await?;
    }
    client.delete(&state.db_pool).await?;
    Ok(StatusCode::OK)
}

#[tracing::instrument(level = "trace", skip(req))]
#[axum_macros::debug_handler]
async fn handle_update_request(
    State(state): State<AppState>,
    Extension(mut client): Extension<ApiClient>,
    req: Json<ApiClientUpdateRequest>,
) -> ModelResult<StatusCode> {
    if let Some(id) = &req.id {
        if id != &client.token.id {
            client.has_permission(&Permission::UPDATE_OTHER)?;
            client = ApiClient::with_id(id.as_str(), &state.db_pool)
                .await
                .map_err(|_| runner!(StatusCode::NOT_FOUND, "Failed to find client by ID"))?;
        }
    } else {
        client.has_permission(&Permission::UPDATE_SELF)?;
    }

    client
        .update(
            &req.name,
            &req.permissions.iter().cloned().collect::<Permission>(),
            &state.db_pool,
        )
        .await?;
    Ok(StatusCode::OK)
}

#[tracing::instrument(level = "trace", skip())]
#[axum_macros::debug_handler]
async fn handle_raw_request(
    Json(req): Json<RawRequest>,
) -> ModelResult<(StatusCode, Json<RawResponse>)> {
    match req.model.as_str() {
        "phi2" => Ok((StatusCode::OK, Json(PHI2_MODEL.clone().run_raw(req)?))),
        "phi3" => Ok((StatusCode::OK, Json(PHI3_MODEL.clone().run_raw(req)?))),
        "mistral7b" => Ok((
            StatusCode::OK,
            Json(MISTRAL7B_INSTRUCT_MODEL.clone().run_raw(req)?),
        )),
        "openhermes" => Ok((StatusCode::OK, Json(OPENHERMES_MODEL.clone().run_raw(req)?))),
        "stablelm2zephyr" => Ok((
            StatusCode::OK,
            Json(STABLELM2_ZEPHYR_MODEL.clone().run_raw(req)?),
        )),
        "stablelm2" => Ok((StatusCode::OK, Json(STABLELM2_MODEL.clone().run_raw(req)?))),
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}

#[tracing::instrument(level = "trace", skip())]
#[axum_macros::debug_handler]
async fn handle_instruct_request(
    Json(req): Json<InstructRequest>,
) -> ModelResult<(StatusCode, Json<InstructResponse>)> {
    match req.model.as_str() {
        "phi2" => Ok((StatusCode::OK, Json(PHI2_MODEL.clone().run_instruct(req)?))),
        "phi3" => Ok((StatusCode::OK, Json(PHI3_MODEL.clone().run_instruct(req)?))),
        "mistral7b" => Ok((
            StatusCode::OK,
            Json(MISTRAL7B_INSTRUCT_MODEL.clone().run_instruct(req)?),
        )),
        "openhermes" => Ok((
            StatusCode::OK,
            Json(OPENHERMES_MODEL.clone().run_instruct(req)?),
        )),
        "stablelm2zephyr" => Ok((
            StatusCode::OK,
            Json(STABLELM2_ZEPHYR_MODEL.clone().run_instruct(req)?),
        )),
        "stablelm2" => Ok((
            StatusCode::OK,
            Json(STABLELM2_MODEL.clone().run_instruct(req)?),
        )),
        _ => bail_runner!(StatusCode::NOT_FOUND, "Model {} not found", req.model),
    }
}

#[tracing::instrument(level = "trace", skip(multipart))]
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
                    )?);
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

/// As per <https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Containers#wave_wav/>
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
