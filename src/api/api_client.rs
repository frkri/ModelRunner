use std::fmt::Display;
use std::time::SystemTime;

use anyhow::bail;
use anyhow::Result;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokio::try_join;

#[allow(dead_code)]
#[derive(Serialize, Debug)]
pub(crate) struct ApiClient {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    #[serde(skip_serializing)]
    pub(crate) key: String,
    pub(crate) permissions: Vec<Permission>,
    pub(crate) created_at: i64,
    pub(crate) updated_at: i64,
    pub(crate) created_by: Option<String>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct ApiClientStatusRequest {
    pub(crate) id: String,
}

#[derive(Deserialize, Debug)]
pub(crate) struct ApiClientCreateRequest {
    pub(crate) name: String,
    pub(crate) permissions: Vec<Permission>,
}

#[derive(Serialize, Debug)]
pub(crate) struct ApiClientCreateResponse {
    pub(crate) key: String,
}

#[derive(Deserialize, Debug)]
pub(crate) struct ApiClientDeleteRequest {
    pub(crate) id: String,
}

#[derive(Deserialize, Debug)]
pub(crate) struct ApiClientUpdateRequest {
    pub(crate) id: Option<String>,
    pub(crate) name: String,
    pub(crate) permissions: Vec<Permission>,
}

#[derive(PartialEq, Deserialize, Serialize, Clone, Copy, Debug, ValueEnum)]
pub enum Permission {
    Use,
    Status,
    Create,
    Delete,
    Update,
}

impl Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Permission::Use => write!(f, "use"),
            Permission::Status => write!(f, "status"),
            Permission::Create => write!(f, "create"),
            Permission::Delete => write!(f, "delete"),
            Permission::Update => write!(f, "update"),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<i64> for Permission {
    fn into(self) -> i64 {
        match self {
            Permission::Use => 1,
            Permission::Status => 2,
            Permission::Create => 3,
            Permission::Delete => 4,
            Permission::Update => 5,
        }
    }
}

impl From<i64> for Permission {
    fn from(item: i64) -> Self {
        match item {
            1 => Permission::Use,
            2 => Permission::Status,
            3 => Permission::Create,
            4 => Permission::Delete,
            5 => Permission::Update,
            _ => unreachable!(),
        }
    }
}

// TODO: Find alternative?
#[allow(clippy::from_over_into)]
impl Into<i64> for &Permission {
    fn into(self) -> i64 {
        match self {
            Permission::Use => 1,
            Permission::Status => 2,
            Permission::Create => 3,
            Permission::Delete => 4,
            Permission::Update => 5,
        }
    }
}

impl ApiClient {
    #[tracing::instrument(level = "info", skip(pool))]
    pub(crate) async fn from(id: &str, pool: &SqlitePool) -> Result<Self> {
        let half_client = sqlx::query!(
            "SELECT id, name, key, created_at, updated_at, created_by FROM api_clients WHERE id = ?",
            id
        )
        .fetch_one(pool);
        let client_permissions = sqlx::query!(
            "SELECT scope_id FROM api_client_permission_scopes WHERE api_client_id = ?",
            id
        )
        .fetch_all(pool);

        let (half_client, client_permissions) = try_join!(half_client, client_permissions)?;
        let permissions: Vec<Permission> = client_permissions
            .iter()
            .map(|p| Permission::from(p.scope_id))
            .collect();
        let client = ApiClient {
            id: half_client.id,
            name: half_client.name,
            key: half_client.key,
            created_at: half_client.created_at,
            updated_at: half_client.updated_at,
            created_by: half_client.created_by,
            permissions,
        };

        Ok(client)
    }
    #[tracing::instrument(level = "info", skip(self))]
    pub(crate) fn has_permission(&self, permission: Permission) -> Result<()> {
        if !self.permissions.contains(&permission) {
            bail!(
                "Client does not have permission to perform this action: {}",
                permission
            )
        }
        Ok(())
    }

    #[tracing::instrument(level = "info", skip_all)]
    pub(crate) async fn delete(&self, pool: &SqlitePool) -> Result<()> {
        sqlx::query!(
            "DELETE FROM api_client_permission_scopes where api_client_id = ?",
            self.id
        )
        .execute(pool)
        .await?;
        sqlx::query!("DELETE FROM api_clients WHERE id = ?", self.id)
            .execute(pool)
            .await?;

        Ok(())
    }

    #[tracing::instrument(level = "info", skip(self, pool))]
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
            "UPDATE api_clients SET name = ?, updated_at = ? WHERE id = ?",
            name,
            unix_now,
            self.id
        )
        .execute(pool);
        let delete_query = sqlx::query!(
            "DELETE FROM api_client_permission_scopes WHERE api_client_id = ?",
            self.id
        )
        .execute(pool);
        try_join!(update_query, delete_query)?;

        for p in permission {
            let scope_id: i64 = p.into();
            sqlx::query!(
                "INSERT INTO api_client_permission_scopes (api_client_id, scope_id) VALUES (?, ?)",
                self.id,
                scope_id
            )
            .execute(pool)
            .await?;
        }

        Ok(())
    }
}
