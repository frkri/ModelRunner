use anyhow::Result;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rand::random;

use crate::inference::model_config::GeneralModelConfig;
use crate::inference::models::model::ModelBase;
use crate::inference::task::instruct::{InstructHandler, InstructRequest, InstructResponse};
use crate::inference::task::raw::{RawHandler, RawRequest, RawResponse};
use crate::inference::text_pipeline::TextGeneratorPipeline;

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs
pub struct OpenHermesModel {
    generator_pipeline: TextGeneratorPipeline,
}

impl Clone for OpenHermesModel {
    fn clone(&self) -> Self {
        OpenHermesModel {
            generator_pipeline: self.generator_pipeline.clone(),
        }
    }
}

impl OpenHermesModel {
    pub fn new(
        api: &Api,
        base: ModelBase,
        tokenizer_filename: &str,
        gguf_filename: &str,
        general_model_config: GeneralModelConfig,
    ) -> Result<Self> {
        let repo = api.repo(Repo::with_revision(
            base.repo_id,
            RepoType::Model,
            base.repo_revision.clone(),
        ));
        let mistral_repo = api.repo(Repo::with_revision(
            "mistralai/Mistral-7B-Instruct-v0.1".into(),
            RepoType::Model,
            "main".into(),
        ));
        let tokenizer_file = mistral_repo.get(tokenizer_filename)?;

        let generator_pipeline = TextGeneratorPipeline::with_quantized_gguf(
            &repo,
            tokenizer_file,
            gguf_filename,
            general_model_config.seed,
            general_model_config.temperature,
            general_model_config.top_p,
            general_model_config.repeat_penalty,
            general_model_config.repeat_context_size,
        )?;

        Ok(OpenHermesModel { generator_pipeline })
    }
}

impl RawHandler for OpenHermesModel {
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

impl InstructHandler for OpenHermesModel {
    fn run_instruct(&mut self, request: InstructRequest) -> Result<InstructResponse> {
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            request.input
        );
        let (output, inference_time) = self
            .generator_pipeline
            .generate(&prompt, request.max_length)?;

        Ok(InstructResponse {
            output,
            inference_time,
        })
    }
}
