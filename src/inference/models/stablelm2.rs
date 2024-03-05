use anyhow::Result;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::stable_lm::Config;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::random;

use crate::inference::model_config::GeneralModelConfig;
use crate::inference::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::inference::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::inference::text_pipeline::{Model, ModelConfig, TextGeneratorPipeline};
use crate::ModelBase;

pub struct StableLm2Model {
    generator_pipeline: TextGeneratorPipeline,
}

impl Clone for StableLm2Model {
    fn clone(&self) -> Self {
        StableLm2Model {
            generator_pipeline: self.generator_pipeline.clone(),
        }
    }
}

impl StableLm2Model {
    pub fn new(
        api: Api,
        base: ModelBase,
        tokenizer_filename: String,
        gguf_filename: String,
        general_model_config: GeneralModelConfig,
    ) -> Result<Self> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id,
            RepoType::Model,
            base.repo_revision,
        ));
        let stablelm_repo = api.repo(Repo::with_revision(
            "stabilityai/stablelm-2-zephyr-1_6b".into(),
            RepoType::Model,
            "095f80bcaa049c014925fec31d145754751899e6".into(), // TODO change later
        ));
        let config = std::fs::read_to_string(stablelm_repo.get("config.json")?)?;
        let config: Config = serde_json::from_str(&config)?;

        let generator_pipeline = TextGeneratorPipeline::with_quantized_gguf_config(
            repo,
            Model::StableLm(None),
            ModelConfig::StableLm(config),
            tokenizer_filename.as_str(),
            gguf_filename.as_str(),
            general_model_config.seed,
            general_model_config.temperature,
            general_model_config.top_p,
            general_model_config.repeat_penalty,
            general_model_config.repeat_context_size,
        )?;

        Ok(StableLm2Model { generator_pipeline })
    }
}

impl RawHandler for StableLm2Model {
    fn run_raw(&mut self, request: RawRequest) -> Result<RawResponse> {
        let pipeline = &mut self.generator_pipeline;
        let logits = LogitsProcessor::new(
            request.model_config.seed.unwrap_or(random()),
            request.model_config.temperature,
            request.model_config.top_p,
        );

        pipeline.repeat_penalty = request.model_config.repeat_penalty;
        pipeline.repeat_context_size = request.model_config.repeat_context_size;
        pipeline.logits_processor = logits;

        let (output, inference_time) = pipeline.generate(&request.input, request.max_length)?;
        Ok(RawResponse {
            output,
            inference_time,
        })
    }
}

impl InstructHandler for StableLm2Model {
    fn run_instruct(&mut self, request: InstructRequest) -> Result<InstructResponse> {
        let prompt = format!("<|user|>\n{}<|endoftext|>\n<|assistant|>\n", request.input);
        let (output, inference_time) = self
            .generator_pipeline
            .generate(&prompt, request.max_length)?;

        Ok(InstructResponse {
            output,
            inference_time,
        })
    }
}
