use std::fmt::{Display, Formatter};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

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
        Self { error: message }
    }
}

impl From<&str> for HttpErrorResponse {
    fn from(message: &str) -> Self {
        Self {
            error: message.to_string(),
        }
    }
}

impl Display for ModelRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message.error)
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
        Self {
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
    ($status_code:expr, $fmt:expr $(, $arg:expr)*) => {
        return Err(ModelRunnerError {
            status: $status_code,
            message: HttpErrorResponse::from(format!($fmt $(, $arg)*)),
        })
    };
}

#[macro_export]
macro_rules! runner {
    ($error_message:expr) => {
        $crate::error::ModelRunnerError { status: StatusCode::INTERNAL_SERVER_ERROR, message: HttpErrorResponse::from($error_message) }
    };
    ($status_code:expr, $error_message:expr) => {
        $crate::error::ModelRunnerError { status: $status_code, message: HttpErrorResponse::from($error_message) }
    };
    ($status_code:expr, $fmt:expr $(, $arg:expr)*) => {
        ModelRunnerError {
            status: $status_code,
            message: HttpErrorResponse::from(format!($fmt $(, $arg)*)),
        }
    };
}
