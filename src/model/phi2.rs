use anyhow::Error;
use candle_transformers::models::mixformer;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::random;

use crate::model::model::ModelBase;
use crate::model::runner::GeneratorPipeline;
use crate::model::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::model::task::raw::{RawHandler, RawRequest, RawResponse};

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/phi/main.rs
pub struct Phi2Model {
    base: ModelBase,
    generator_pipeline: GeneratorPipeline,
}

pub struct Phi2ModelConfig {
    seed: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_context_size: usize,
}

impl Default for Phi2ModelConfig {
    fn default() -> Self {
        Phi2ModelConfig {
            seed: random(),
            temperature: Some(0.8),
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_context_size: 64,
        }
    }
}

impl Phi2Model {
    pub fn new(
        api: Api,
        base: ModelBase,
        tokenizer_filename: String,
        gguf_filename: String,
        mixformer_config: mixformer::Config,
        phi2_config: Phi2ModelConfig,
    ) -> Result<Self, Error> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id.clone(),
            RepoType::Model,
            base.repo_revision.clone(),
        ));

        let generator_pipeline = GeneratorPipeline::with_gguf_mixformer_model(
            repo,
            mixformer_config,
            tokenizer_filename.as_str(),
            gguf_filename.as_str(),
            phi2_config.seed,
            phi2_config.temperature,
            phi2_config.top_p,
            phi2_config.repeat_penalty,
            phi2_config.repeat_context_size,
        )?;

        let model = Phi2Model {
            base,
            generator_pipeline,
        };

        Ok(model)
    }
}

impl RawHandler for Phi2Model {
    fn run(&mut self, params: RawRequest) -> Result<RawResponse, Error> {
        let (output, inference_time) = self
            .generator_pipeline
            .generate(&params.input, params.max_length)?;
        Ok(RawResponse {
            output,
            inference_time,
        })
    }
}

impl InstructHandler for Phi2Model {
    fn run(&mut self, params: InstructRequest) -> Result<InstructResponse, Error> {
        let prompt = format!("USER: {}\nASSISTANT:", params.input);
        let (output, inference_time) = self
            .generator_pipeline
            .generate(&prompt, params.max_length)?;

        Ok(InstructResponse {
            output,
            inference_time,
        })
    }
}
