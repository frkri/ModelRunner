use std::fmt::Display;
use std::str::FromStr;
use std::time::SystemTime;

use anyhow::Result;
use anyhow::{anyhow, bail};
use bitflags::bitflags;
use password_hash::rand_core::OsRng;
use password_hash::{PasswordHash, PasswordVerifier, SaltString};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::api::auth::{Auth, AuthToken};

#[allow(dead_code)]
#[derive(Serialize, Clone)]
pub struct ApiClient {
    pub name: Option<String>,
    pub token: AuthToken,
    pub permissions: Permission,
    pub created_at: i64,
    pub updated_at: i64,
    pub created_by: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct ApiClientStatusRequest {
    pub(crate) id: String,
}

#[derive(Deserialize)]
pub(crate) struct ApiClientCreateRequest {
    pub(crate) name: String,
    pub(crate) permissions: Vec<Permission>,
}

#[derive(Deserialize)]
pub(crate) struct ApiClientDeleteRequest {
    pub(crate) id: String,
}

#[derive(Deserialize)]
pub(crate) struct ApiClientUpdateRequest {
    pub(crate) id: Option<String>,
    pub(crate) name: String,
    pub(crate) permissions: Vec<Permission>,
}

bitflags! {
    // i64 is used to store the bitflags in the sqlite db
    #[derive(Serialize, Deserialize, Clone, Debug)]
    #[serde(transparent)]
    pub struct Permission: i64 {
        const USE_SELF        = 1 << 0;
        const USE_OTHER       = 1 << 1;
        const STATUS_SELF     = 1 << 2;
        const STATUS_OTHER    = 1 << 3;
        const CREATE_SELF     = 1 << 4;
        const CREATE_OTHER    = 1 << 5;
        const DELETE_SELF     = 1 << 6;
        const DELETE_OTHER    = 1 << 7;
        const UPDATE_SELF     = 1 << 8;
        const UPDATE_OTHER    = 1 << 9;
    }
}

impl FromStr for Permission {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Permission::from_name(s).ok_or_else(|| anyhow!("Invalid permission"))
    }
}

impl Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

impl Display for ApiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Name: {}\nToken: {}\nPermissions: {:?}\nCreated At: {}\nUpdated At: {}\nCreated By: {:?}",
            self.name.as_ref().unwrap_or(&"None".to_string()),
            self.token,
            self.permissions,
            self.created_at,
            self.updated_at,
            self.created_by.as_ref().unwrap_or(&"None".to_string())
        )
    }
}

impl ApiClient {
    pub async fn new(
        auth: &Auth,
        name: &str,
        permission: &Permission,
        creator_id: &Option<String>,
        pool: &SqlitePool,
    ) -> Result<ApiClient> {
        let salt = SaltString::generate(&mut OsRng);
        let token = AuthToken::new(&auth.argon, &salt)?;
        let key_hash = &token
            .key_hash
            .as_ref()
            .ok_or_else(|| anyhow!("Hash not found"))?
            .to_string();

        let unix_now: i64 = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_millis()
            .try_into()?;
        let permission_bits: i64 = permission.bits();
        sqlx::query!(
            "INSERT INTO client (id, name, key, permissions, created_at, updated_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?)",
            token.id,
            name,
            key_hash,
            permission_bits,
            unix_now,
            unix_now,
            creator_id
        )
            .execute(pool)
            .await?;

        Ok(ApiClient {
            name: Some(name.to_string()),
            token,
            permissions: permission.to_owned(),
            created_at: unix_now,
            updated_at: unix_now,
            created_by: creator_id.clone(),
        })
    }

    pub(crate) async fn with_id(id: &str, pool: &SqlitePool) -> Result<Self> {
        let client_record = sqlx::query!(
            "SELECT id, name, key, permissions, created_at, updated_at, created_by FROM client WHERE id = ?",
            id
        )
            .fetch_one(pool).await?;

        Ok(ApiClient {
            name: client_record.name,
            token: AuthToken::from(
                client_record.id,
                PasswordHash::new(client_record.key.as_str()).map_err(|e| anyhow!(e))?,
            ),
            created_at: client_record.created_at,
            updated_at: client_record.updated_at,
            created_by: client_record.created_by,
            permissions: Permission::from_bits(client_record.permissions)
                .ok_or_else(|| anyhow!("Permission not found"))?,
        })
    }

    pub(crate) async fn with_token(
        auth: &Auth,
        token: AuthToken,
        pool: &SqlitePool,
    ) -> Result<Self> {
        let client_record = sqlx::query!(
            "SELECT id, name, key, permissions, created_at, updated_at, created_by FROM client WHERE id = ?",
            token.id
        )
            .fetch_one(pool).await?;
        let stored_hashed_key =
            PasswordHash::new(client_record.key.as_str()).map_err(|e| anyhow!(e))?;
        let key = token
            .key_raw
            .ok_or_else(|| anyhow!("Token key not found"))?;
        auth.argon
            .verify_password(key.as_bytes(), &stored_hashed_key)
            .map_err(|e| anyhow!(e))?;

        let client = ApiClient {
            name: client_record.name,
            token: AuthToken::from(client_record.id, stored_hashed_key),
            created_at: client_record.created_at,
            updated_at: client_record.updated_at,
            created_by: client_record.created_by,
            permissions: Permission::from_bits(client_record.permissions)
                .ok_or_else(|| anyhow!("Permission not found"))?,
        };

        Ok(client)
    }
    pub(crate) fn has_permission(&self, permission: &Permission) -> Result<()> {
        if !self.permissions.contains(permission.to_owned()) {
            bail!(
                "Client does not have permission to perform this action: {:?}",
                permission
            )
        }
        Ok(())
    }

    pub(crate) async fn delete(&self, pool: &SqlitePool) -> Result<()> {
        sqlx::query!("DELETE FROM client WHERE id = ?", self.token.id)
            .execute(pool)
            .await?;
        Ok(())
    }

    pub(crate) async fn update(
        &self,
        name: &String,
        permission: &Permission,
        pool: &SqlitePool,
    ) -> Result<()> {
        let unix_now: i64 = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_millis()
            .try_into()?;
        let permission_bits = permission.bits();
        sqlx::query!(
            "UPDATE client SET name = ?, permissions = ?, updated_at = ? WHERE id = ?",
            name,
            permission_bits,
            unix_now,
            self.token.id
        )
        .execute(pool)
        .await?;

        Ok(())
    }
}
