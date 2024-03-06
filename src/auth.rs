use std::hint::black_box;

use anyhow::{anyhow, Result};
use argon2::Argon2;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use base64ct::Base64;
use base64ct::Encoding;
use password_hash::rand_core::OsRng;
use password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use rand::RngCore;
use sqlx::{SqlitePool};

use crate::error::ModelResult;
use crate::HttpErrorResponse;
use crate::{bail_runner, AppState};

#[allow(clippy::unit_arg)]
async fn check_api_key(key: &str, pool: &SqlitePool) -> Result<bool> {
    let clients = sqlx::query_as!(ApiClient, "SELECT id, name, key FROM api_clients")
        .fetch_all(pool)
        .await?;

    let mut is_valid = false;
    black_box(for client in clients {
        let hashed_key = PasswordHash::new(&client.key).map_err(|e| anyhow!(e))?;
        if Argon2::default()
            .verify_password(key.as_bytes(), &hashed_key)
            .is_ok()
        {
            is_valid = true
        }
    });

    Ok(is_valid)
}

pub(crate) async fn create_api_key(name: &str, pool: &SqlitePool) -> Result<String> {
    let mut key = [0u8; 32];
    OsRng.fill_bytes(&mut key);
    let key = Base64::encode_string(&key);
    let salt = SaltString::generate(&mut OsRng);

    let hash = Argon2::default()
        .hash_password(key.as_bytes(), &salt)
        .map_err(|e| anyhow!(e))?
        .to_string();
    sqlx::query!(
        "INSERT INTO api_clients (name, key) VALUES (?, ?)",
        name,
        hash
    )
    .execute(pool)
    .await?;

    Ok(key)
}

pub(crate) async fn auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> ModelResult<Response> {
    let header = request.headers().get("authorization");

    let key = match header {
        Some(key) => {
            let str = key.to_str()?;
            if !str.starts_with("Bearer ") {
                bail_runner!(StatusCode::BAD_REQUEST, "Invalid authorization header")
            }
            str.trim_start_matches("Bearer ")
        }
        None => bail_runner!(StatusCode::BAD_REQUEST, "Missing authorization header"),
    };
    if check_api_key(&key, &state.db_pool).await? {
        Ok(next.run(request).await)
    } else {
        bail_runner!(StatusCode::UNAUTHORIZED, "Invalid API key")
    }
}
