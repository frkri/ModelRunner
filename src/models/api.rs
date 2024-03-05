#[derive(sqlx::FromRow)]
pub(crate) struct ApiClient {
    pub(crate) id: i64,
    pub(crate) name: Option<String>,
    pub(crate) key: String,
}
