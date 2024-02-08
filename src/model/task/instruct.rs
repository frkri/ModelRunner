use anyhow::Error;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct InstructRequest {
    pub model: String,
    pub input: String,
    pub max_length: usize,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct InstructResponse {
    pub output: String,
    pub inference_time: f64,
}

pub trait InstructHandler {
    fn run(&mut self, params: InstructRequest) -> Result<InstructResponse, Error>;
}
