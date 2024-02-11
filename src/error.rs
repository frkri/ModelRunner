use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

// Taken from https://github.com/tokio-rs/axum/blob/main/examples/anyhow-error-response/src/main.rs
#[derive(Debug)]
pub struct ModelRunnerError {
    pub status: StatusCode,
    pub message: HttpErrorResponse,
}

#[derive(Debug, Serialize)]
pub struct HttpErrorResponse {
    error: String,
}

impl From<String> for HttpErrorResponse {
    fn from(message: String) -> Self {
        HttpErrorResponse { error: message }
    }
}

impl From<&str> for HttpErrorResponse {
    fn from(message: &str) -> Self {
        HttpErrorResponse {
            error: message.to_string(),
        }
    }
}

impl IntoResponse for ModelRunnerError {
    fn into_response(self) -> Response {
        let mut res = Json(self.message).into_response();
        *res.status_mut() = self.status;
        res
    }
}

impl<E> From<E> for ModelRunnerError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        ModelRunnerError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: HttpErrorResponse::from(err.into().to_string()),
        }
    }
}

pub type ModelResult<T, E = ModelRunnerError> = Result<T, E>;

#[macro_export]
macro_rules! bail_runner {
    ($error_message:expr) => {
        return Err($crate::error::ModelRunnerError { status: StatusCode::INTERNAL_SERVER_ERROR, message: HttpErrorResponse::from($error_message) })
    };
    ($status_code:expr, $error_message:expr) => {
        return Err($crate::error::ModelRunnerError { status: $status_code, message: HttpErrorResponse::from($error_message) })
    };
    ($status:expr, $fmt:expr $(, $arg:expr)*) => {
        return Err(ModelRunnerError {
            status: $status,
            message: HttpErrorResponse::from(format!($fmt $(, $arg)*)),
        })
    };
}
