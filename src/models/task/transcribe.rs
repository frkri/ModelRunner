use anyhow::Error;
use serde::{Deserialize, Serialize};

use crate::models::runner::Segment;

#[derive(Deserialize, Debug)]
pub struct TranscribeRequest {
    pub model: String,
    pub language: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TranscribeResponse {
    pub output: Vec<Segment>,
    pub inference_time: f64,
}

pub trait TranscribeHandler {
    fn run_transcribe(
        &mut self,
        input: Box<[u8]>,
        language_token: &String,
    ) -> Result<TranscribeResponse, Error>;
}
