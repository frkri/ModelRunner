use anyhow::Error;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct InstructRequest {
    pub model: String,
    pub input: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct InstructResponse {
    pub output: String,
}

pub trait InstructHandler {
    fn run(&self, params: InstructRequest) -> Result<InstructResponse, Error>;
}
