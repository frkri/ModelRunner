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

#[derive(Clone)]
pub struct StableLm2Model {
    generator_pipeline: TextGeneratorPipeline,
    insert_prompt: bool,
}

impl StableLm2Model {
    #[tracing::instrument(level = "info", skip(api))]
    pub fn new(
        api: &Api,
        base: &ModelBase,
        tokenizer_filename: &str,
        gguf_filename: &str,
        general_model_config: &GeneralModelConfig,
        insert_prompt: bool,
    ) -> Result<Self> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id.clone(),
            RepoType::Model,
            base.repo_revision.clone(),
        ));
        let stablelm_repo = api.repo(Repo::with_revision(
            "stabilityai/stablelm-2-zephyr-1_6b".into(),
            RepoType::Model,
            "main".into(),
        ));
        let config = std::fs::read_to_string(stablelm_repo.get("config.json")?)?;
        let config: Config = serde_json::from_str(&config)?;

        let generator_pipeline = TextGeneratorPipeline::with_quantized_gguf_config(
            &repo,
            &Model::StableLm(None),
            ModelConfig::StableLm(config),
            tokenizer_filename,
            gguf_filename,
            general_model_config.seed,
            general_model_config.temperature,
            general_model_config.top_p,
            general_model_config.repeat_penalty,
            general_model_config.repeat_context_size,
        )?;

        Ok(Self {
            generator_pipeline,
            insert_prompt,
        })
    }
}

impl RawHandler for StableLm2Model {
    #[tracing::instrument(level = "info", skip(self, request))]
    fn run_raw(&mut self, request: RawRequest) -> Result<RawResponse> {
        let pipeline = &mut self.generator_pipeline;
        let logits = LogitsProcessor::new(
            request.model_config.seed.unwrap_or_else(random),
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
    #[tracing::instrument(level = "info", skip(self, request))]
    fn run_instruct(&mut self, request: InstructRequest) -> Result<InstructResponse> {
        let prompt = if self.insert_prompt {
            format!("<|user|>\n{}<|endoftext|>\n<|assistant|>\n", request.input)
        } else {
            request.input
        };
        let (output, inference_time) = self
            .generator_pipeline
            .generate(&prompt, request.max_length)?;

        Ok(InstructResponse {
            output,
            inference_time,
        })
    }
}
