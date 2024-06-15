use anyhow::Result;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::random;

use crate::inference::model_config::GeneralModelConfig;
use crate::inference::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::inference::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::inference::text_pipeline::{Model, ModelConfig, TextGeneratorPipeline};
use crate::ModelBase;

#[derive(Clone)]
pub struct PhiModel {
    pub base: ModelBase,
    generator_pipeline: TextGeneratorPipeline,
    alt_prompt: bool,
}

impl PhiModel {
    #[tracing::instrument(level = "info", skip(api))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api: &Api,
        base: &ModelBase,
        tokenizer_repo: &str,
        tokenizer_filename: &str,
        gguf_filename: &str,
        phi2_config: Option<mixformer::Config>,
        general_model_config: GeneralModelConfig,
        alt_prompt: bool,
    ) -> Result<Self> {
        let phi_repo = api.repo(Repo::with_revision(
            base.repo_id.clone(),
            RepoType::Model,
            base.repo_revision.clone(),
        ));
        let tokenizer_repo = api.repo(Repo::with_revision(
            tokenizer_repo.into(),
            RepoType::Model,
            "main".into(),
        ));

        let model_type = if alt_prompt {
            Model::Phi3(None)
        } else {
            Model::Phi2(None)
        };
        let generator_pipeline = if phi2_config.is_some() {
            TextGeneratorPipeline::with_quantized_gguf_config(
                &phi_repo,
                &model_type,
                ModelConfig::Phi2(phi2_config.unwrap()),
                tokenizer_filename,
                gguf_filename,
                general_model_config.seed,
                general_model_config.temperature,
                general_model_config.top_p,
                general_model_config.repeat_penalty,
                general_model_config.repeat_context_size,
            )?
        } else {
            let tokenizer_file = tokenizer_repo.get(tokenizer_filename)?;
            TextGeneratorPipeline::with_quantized_gguf(
                &phi_repo,
                &model_type,
                tokenizer_file,
                gguf_filename,
                general_model_config.seed,
                general_model_config.temperature,
                general_model_config.top_p,
                general_model_config.repeat_penalty,
                general_model_config.repeat_context_size,
            )?
        };

        Ok(Self {
            base: base.clone(),
            generator_pipeline,
            alt_prompt,
        })
    }
}

impl RawHandler for PhiModel {
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

impl InstructHandler for PhiModel {
    #[tracing::instrument(level = "info", skip(self, request))]
    fn run_instruct(&mut self, request: InstructRequest) -> Result<InstructResponse> {
        let prompt = if self.alt_prompt {
            format!("<|user|>\n{}<|end|>\n<|assistant|>\n", request.input)
        } else {
            format!("Instruct: {}\nOutput:", request.input)
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
