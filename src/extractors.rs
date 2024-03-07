use axum::async_trait;
use axum::extract::FromRef;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;

use crate::auth::{extract_auth_header, extract_id_key};
use crate::error::ModelRunnerError;
use crate::models::api::ApiClient;
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

        let client = ApiClient::from(id, &AppState::from_ref(state).db_pool).await?;
        Ok(ApiClientExtractor(client))
    }
}
