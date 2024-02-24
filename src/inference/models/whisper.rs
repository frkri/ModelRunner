use anyhow::{Error, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::SeedableRng;

use crate::inference::audio_pipeline::AudioGeneratorPipeline;
use crate::inference::models::model::ModelBase;
use crate::inference::task::transcribe::{TranscribeHandler, TranscribeResponse};

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/whisper/main.rs
pub struct WhisperModel {
    api: Api,
    generator_pipeline: AudioGeneratorPipeline,
}

impl Clone for WhisperModel {
    fn clone(&self) -> Self {
        WhisperModel {
            api: self.api.clone(),
            generator_pipeline: self.generator_pipeline.clone(),
        }
    }
}

impl WhisperModel {
    pub fn new(
        api: Api,
        base: ModelBase,
        config_filename: String,
        tokenizer_filename: String,
        gguf_filename: String,
        mel_filters_filename: String,
    ) -> Result<Self> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id.clone(),
            RepoType::Model,
            base.repo_revision.clone(),
        ));
        let generator_pipeline = AudioGeneratorPipeline::with_gguf_model(
            repo,
            config_filename.as_str(),
            tokenizer_filename.as_str(),
            gguf_filename.as_str(),
            mel_filters_filename.as_str(),
            true,
            rand::rngs::StdRng::from_seed([0; 32]),
        )?;

        Ok(Self {
            api,
            generator_pipeline,
        })
    }
}

impl TranscribeHandler for WhisperModel {
    fn run_transcribe(
        &mut self,
        input: Box<[u8]>,
        language_token: &str,
    ) -> Result<TranscribeResponse, Error> {
        let output = self.generator_pipeline.transcribe(input, language_token)?;

        Ok(TranscribeResponse {
            output,
            inference_time: 0.0,
        })
    }
}
