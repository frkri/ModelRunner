use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct InfoRequest {
    pub model: String,
}
