use anyhow::Error;
use serde::{Deserialize, Serialize};

use crate::GeneralModelConfig;

#[derive(Deserialize, Debug)]
pub struct RawRequest {
    pub model: String,
    pub input: String,
    pub max_length: usize,
    pub model_config: GeneralModelConfig,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct RawResponse {
    pub output: String,
    pub inference_time: f64,
}

pub trait RawHandler {
    fn run_raw(&mut self, params: RawRequest) -> Result<RawResponse, Error>;
}
