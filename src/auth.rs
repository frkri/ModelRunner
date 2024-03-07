use anyhow::{anyhow, Result};
use argon2::Argon2;
use base64ct::Base64;
use base64ct::Encoding;
use password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use rand::rngs::OsRng;
use rand::RngCore;
use sqlx::SqlitePool;

#[derive(Clone)]
pub(crate) struct Auth {
    argon: Argon2<'static>,
}

impl Default for Auth {
    fn default() -> Self {
        let argon = Argon2::default();
        Self { argon }
    }
}

impl Auth {
    pub(crate) async fn create_api_key(&self, name: &str, pool: &SqlitePool) -> Result<String> {
        let mut key = [0u8; 64];
        OsRng.fill_bytes(&mut key);
        let key = Base64::encode_string(&key);

        let mut id = [0u8; 16];
        OsRng.fill_bytes(&mut id);
        let id = Base64::encode_string(&id);

        let salt = SaltString::generate(&mut OsRng);
        let key_hash = self
            .argon
            .hash_password(key.as_bytes(), &salt)
            .map_err(|e| anyhow!(e))?
            .to_string();

        sqlx::query_as!(
            ApiClient,
            "INSERT INTO api_clients (id, name, key) VALUES (?, ?, ?)",
            id,
            name,
            key_hash
        )
        .execute(pool)
        .await?;

        Ok(format!("{}_{}", id, key))
    }

    pub(crate) async fn check_api_key(&self, key: &str, pool: &SqlitePool) -> Result<bool> {
        let mut parts = key.split('_');
        let id = parts.next().ok_or(anyhow!("Invalid format for key"))?;
        let key = parts.next().ok_or(anyhow!("Invalid format for key"))?;

        let hashed_key_record = sqlx::query!("SELECT key FROM api_clients WHERE id = ?", id)
            .fetch_one(pool)
            .await
            .map_err(|_| anyhow!("Invalid key"))?;
        let hashed_key =
            PasswordHash::new(hashed_key_record.key.as_str()).map_err(|e| anyhow!(e))?;

        let is_valid = self
            .argon
            .verify_password(key.as_bytes(), &hashed_key)
            .is_ok();
        Ok(is_valid)
    }
}
