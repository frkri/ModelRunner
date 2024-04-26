use std::fmt::Display;

use anyhow::Result;
use anyhow::{anyhow, bail};
use argon2::Argon2;
use base64ct::Base64;
use base64ct::Encoding;
use password_hash::rand_core::OsRng;
use password_hash::{PasswordHashString, PasswordHasher, SaltString};
use rand::RngCore;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct Auth {
    pub(crate) argon: Argon2<'static>,
}

impl Default for Auth {
    #[tracing::instrument(level = "trace")]
    fn default() -> Self {
        let argon = Argon2::default();
        Self { argon }
    }
}

const AUTH_TOKEN_SEPARATOR: &str = "_";

/// `AuthToken` is a struct that holds the id and a hashed key of a token. It also provides the display format of the token which is delimited by `AUTH_TOKEN_SEPARATOR`.
#[derive(Serialize, Clone, Debug)]
pub struct AuthToken {
    pub id: String,
    #[serde(skip)]
    pub key_hash: Option<PasswordHashString>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_raw: Option<String>,
}

impl AuthToken {
    #[tracing::instrument(level = "info", skip(argon, salt))]
    pub(crate) fn new(argon: &Argon2, salt: &SaltString) -> Result<Self> {
        let mut key = [0u8; 64];
        OsRng.fill_bytes(&mut key);
        let key = Base64::encode_string(&key);
        let key = key.trim_end_matches('=');

        let mut id = [0u8; 16];
        OsRng.fill_bytes(&mut id);
        let id = Base64::encode_string(&id);
        let id = id.trim_end_matches('=').to_string();

        let key_hash = Some(
            argon
                .hash_password(key.as_bytes(), salt)
                .map_err(|e| anyhow!(e))?
                .into(),
        );

        Ok(Self {
            id,
            key_hash,
            key_raw: Some(key.to_string()),
        })
    }

    #[tracing::instrument(level = "info", skip(hash))]
    pub(crate) fn from(id: String, hash: impl Into<PasswordHashString>) -> Self {
        let hash = hash.into();
        Self {
            id,
            key_hash: Some(hash),
            key_raw: None,
        }
    }

    #[tracing::instrument(level = "info", skip(token))]
    pub(crate) fn from_raw_str(token: &str) -> Result<Self> {
        let parts: Vec<&str> = token.split(AUTH_TOKEN_SEPARATOR).collect();
        if parts.len() != 2 {
            bail!("Invalid token format");
        }
        Ok(Self {
            id: parts[0].to_string(),
            key_hash: None,
            key_raw: Some(parts[1].to_string()),
        })
    }
}

impl Display for AuthToken {
    #[tracing::instrument(level = "trace", skip(self, f))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}",
            self.id,
            AUTH_TOKEN_SEPARATOR,
            self.key_raw.as_ref().unwrap()
        )
    }
}
