use std::fmt::Display;
use std::time::SystemTime;

use anyhow::Result;
use anyhow::{anyhow, bail};
use clap::ValueEnum;
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};
use password_hash::rand_core::OsRng;
use password_hash::{PasswordHash, PasswordVerifier, SaltString};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use strum::Display;
use tokio::try_join;

use crate::api::auth::{Auth, AuthToken};

#[allow(dead_code)]
#[derive(Serialize, Clone)]
pub struct ApiClient {
    pub name: Option<String>,
    pub token: AuthToken,
    pub permissions: Vec<Permission>,
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

#[derive(
    PartialEq, Deserialize, Serialize, Clone, Debug, ValueEnum, Display, FromPrimitive, ToPrimitive,
)]
#[repr(i64)]
/// Permissions are divided between targets Self and Other.
pub enum Permission {
    UseSelf,
    UseOther,
    StatusSelf,
    StatusOther,
    CreateSelf,
    CreateOther,
    DeleteSelf,
    DeleteOther,
    UpdateSelf,
    UpdateOther,
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
        permission: &Vec<Permission>,
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
        sqlx::query_as!(
            ApiClient,
            "INSERT INTO client (id, name, key, created_at, updated_at, created_by) VALUES (?, ?, ?, ?, ?, ?)",
            token.id,
            name,
            key_hash,
            unix_now,
            unix_now,
            creator_id
        )
            .execute(pool)
            .await?;
        for p in permission {
            let permission_id: i64 = p.to_i64().ok_or_else(|| anyhow!("Permission not found"))?;
            sqlx::query!(
                "INSERT INTO client_permission (client_id, permission_id) VALUES (?, ?)",
                token.id,
                permission_id
            )
            .execute(pool)
            .await?;
        }

        Ok(ApiClient {
            name: Some(name.to_string()),
            token,
            permissions: permission.clone(),
            created_at: unix_now,
            updated_at: unix_now,
            created_by: creator_id.clone(),
        })
    }

    pub(crate) async fn with_id(id: &str, pool: &SqlitePool) -> Result<Self> {
        let client_record = sqlx::query!(
            "SELECT id, name, key, created_at, updated_at, created_by FROM client WHERE id = ?",
            id
        )
        .fetch_one(pool);
        let client_permissions = sqlx::query!(
            "SELECT permission_id FROM client_permission WHERE client_id = ?",
            id
        )
        .fetch_all(pool);
        let (client_record, client_permissions) = try_join!(client_record, client_permissions)?;
        let permissions: Vec<Permission> = client_permissions
            .iter()
            .map(|p| {
                Permission::from_i64(p.permission_id)
                    .ok_or_else(|| anyhow!("Permission not found"))
                    .unwrap()
            })
            .collect();

        Ok(ApiClient {
            name: client_record.name,
            token: AuthToken::from(
                client_record.id,
                PasswordHash::new(client_record.key.as_str()).map_err(|e| anyhow!(e))?,
            ),
            created_at: client_record.created_at,
            updated_at: client_record.updated_at,
            created_by: client_record.created_by,
            permissions,
        })
    }

    pub(crate) async fn with_token(
        auth: &Auth,
        token: AuthToken,
        pool: &SqlitePool,
    ) -> Result<Self> {
        let client_record = sqlx::query!(
            "SELECT id, name, key, created_at, updated_at, created_by FROM client WHERE id = ?",
            token.id
        )
        .fetch_one(pool)
        .await?;
        let stored_hashed_key =
            PasswordHash::new(client_record.key.as_str()).map_err(|e| anyhow!(e))?;
        let key = token
            .key_raw
            .ok_or_else(|| anyhow!("Token key not found"))?;
        auth.argon
            .verify_password(key.as_bytes(), &stored_hashed_key)
            .map_err(|e| anyhow!(e))?;

        let client_permissions = sqlx::query!(
            "SELECT permission_id FROM client_permission WHERE client_id = ?",
            token.id
        )
        .fetch_all(pool)
        .await?;
        let permissions: Vec<Permission> = client_permissions
            .iter()
            .map(|p| {
                Permission::from_i64(p.permission_id)
                    .ok_or_else(|| anyhow!("Permission not found"))
                    .unwrap()
            })
            .collect();

        let client = ApiClient {
            name: client_record.name,
            token: AuthToken::from(client_record.id, stored_hashed_key),
            created_at: client_record.created_at,
            updated_at: client_record.updated_at,
            created_by: client_record.created_by,
            permissions,
        };

        Ok(client)
    }
    pub(crate) fn has_permission(&self, permission: Permission) -> Result<()> {
        if !self.permissions.contains(&permission) {
            bail!(
                "Client does not have permission to perform this action: {:?}",
                permission
            )
        }
        Ok(())
    }

    pub(crate) async fn delete(&self, pool: &SqlitePool) -> Result<()> {
        sqlx::query!(
            "DELETE FROM client_permission where client_id = ?",
            self.token.id
        )
        .execute(pool)
        .await?;
        sqlx::query!("DELETE FROM client WHERE id = ?", self.token.id)
            .execute(pool)
            .await?;

        Ok(())
    }

    pub(crate) async fn update(
        &self,
        name: &String,
        permission: &Vec<Permission>,
        pool: &SqlitePool,
    ) -> Result<()> {
        let unix_now: i64 = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_millis()
            .try_into()?;
        let update_query = sqlx::query!(
            "UPDATE client SET name = ?, updated_at = ? WHERE id = ?",
            name,
            unix_now,
            self.token.id
        )
        .execute(pool);
        let delete_query = sqlx::query!(
            "DELETE FROM client_permission WHERE client_id = ?",
            self.token.id
        )
        .execute(pool);
        try_join!(update_query, delete_query)?;

        for p in permission {
            let permission_id: i64 = p.to_i64().ok_or_else(|| anyhow!("Permission not found"))?;
            sqlx::query!(
                "INSERT INTO client_permission (client_id, permission_id) VALUES (?, ?)",
                self.token.id,
                permission_id
            )
            .execute(pool)
            .await?;
        }

        Ok(())
    }
}
