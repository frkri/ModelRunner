use anyhow::Error;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct RawRequest {
    pub model: String,
    pub input: String,
    pub max_length: usize,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct RawResponse {
    pub output: String,
    pub inference_time: f64,
}

pub trait RawHandler {
    fn run(&mut self, params: RawRequest) -> Result<RawResponse, Error>;
}
