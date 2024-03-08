use anyhow::anyhow;
use axum::async_trait;
use axum::extract::FromRef;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;

use crate::api::api_client::ApiClient;
use crate::api::auth::extract_auth_header;
use crate::api::auth::extract_id_key;
use crate::error::ModelRunnerError;
use crate::AppState;

pub(crate) struct ApiClientExtractor(pub(crate) ApiClient);

#[async_trait]
impl<S> FromRequestParts<S> for ApiClientExtractor
where
    AppState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = ModelRunnerError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let header_value = extract_auth_header(&parts.headers)?;
        let (id, _) = extract_id_key(header_value)?;

        let client = ApiClient::from(id, &AppState::from_ref(state).db_pool)
            .await
            .map_err(|_| anyhow!("Failed to find any client matching ID"))?;
        Ok(ApiClientExtractor(client))
    }
}
