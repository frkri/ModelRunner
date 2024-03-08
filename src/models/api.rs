use std::fmt::Display;

use anyhow::bail;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[allow(dead_code)]
#[derive(Serialize)]
pub(crate) struct ApiClient {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    #[serde(skip)]
    pub(crate) key: String,
    pub(crate) permissions: Vec<Permission>,
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

#[derive(Serialize)]
pub(crate) struct ApiClientCreateResponse {
    pub(crate) key: String,
}

#[derive(Deserialize)]
pub(crate) struct ApiClientDeleteRequest {
    pub(crate) id: String,
}

#[derive(PartialEq, Deserialize, Serialize)]
pub(crate) enum Permission {
    Use,
    Status,
    Create,
    Delete,
}

impl Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Permission::Use => write!(f, "use"),
            Permission::Status => write!(f, "status"),
            Permission::Create => write!(f, "create"),
            Permission::Delete => write!(f, "delete"),
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
            _ => unreachable!(),
        }
    }
}

impl ApiClient {
    pub(crate) async fn from(id: &str, pool: &SqlitePool) -> Result<Self> {
        let half_client =
            sqlx::query!("SELECT id, name, key FROM api_clients WHERE id = ?", id).fetch_one(pool);
        let client_permissions = sqlx::query!(
            "SELECT scope_id FROM api_client_permission_scopes WHERE api_client_id = ?",
            id
        )
        .fetch_all(pool);

        let (half_client, client_permissions) = tokio::try_join!(half_client, client_permissions)?;
        let permissions: Vec<Permission> = client_permissions
            .iter()
            .map(|p| Permission::from(p.scope_id))
            .collect();
        let client = ApiClient {
            id: half_client.id,
            name: half_client.name,
            key: half_client.key,
            permissions,
        };

        Ok(client)
    }
    pub(crate) fn has_permission(&self, permission: Permission) -> Result<()> {
        if !self.permissions.contains(&permission) {
            bail!(
                "Client does not have permission to perform this action: {}",
                permission
            )
        }
        Ok(())
    }
}
