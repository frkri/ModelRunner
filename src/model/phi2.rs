use anyhow::{bail, Error};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use candle_transformers::quantized_var_builder;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use log::debug;
use quantized_var_builder::VarBuilder;
use tokenizers::Tokenizer;

use crate::model::task::raw::{RawHandler, RawRequest, RawResponse};

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/phi/main.rs
pub struct Phi2Model {
    generator_pipeline: Option<GeneratorPipeline>,
    unmodified_model: Option<MixFormerSequentialForCausalLM>,
    repo_id: String,
    repo_revision: String,
    weights_file: String,
    seed: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
}
impl RawHandler for Phi2Model {
    fn run(&mut self, params: RawRequest) -> Result<RawResponse, Error> {
        debug!("Running inference on Phi2Model: {:?}", params);

        let (output, inference_time) = self
            .generator_pipeline
            .as_mut()
            .unwrap()
            .generate(&params.input, 150)?;

        Ok(RawResponse {
            output,
            inference_time,
        })
    }
}

impl Phi2Model {
    pub fn new(
        repo_id: String,
        repo_revision: String,
        weights_file: String,
        seed: Option<u64>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<Self, Error> {
        let mut phi2_model = Self {
            generator_pipeline: None,
            unmodified_model: None,
            repo_id,
            repo_revision,
            weights_file,
            seed,
            temperature,
            top_p,
        };

        phi2_model.load_model()?;
        Ok(phi2_model)
    }

    fn load_model(&mut self) -> Result<(), Error> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            self.repo_id.clone(),
            RepoType::Model,
            self.repo_revision.clone(),
        ));

        let tokenizer_file = repo.get("tokenizer-puffin-phi-v2.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();
        let device = Device::Cpu;

        let gguf_file = repo.get(&self.weights_file)?;

        let config = mixformer::Config::puffin_phi_v2();
        let vb = VarBuilder::from_gguf(gguf_file, &device)?;
        let model = MixFormerSequentialForCausalLM::new(&config, vb)?;

        self.unmodified_model = Some(model.clone());
        let pipeline = GeneratorPipeline {
            model,
            device: device.clone(),
            tokenizer,
            logits_processor: LogitsProcessor::new(
                self.seed.unwrap_or(299792458),
                self.temperature,
                self.top_p,
            ),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        };
        self.generator_pipeline = Some(pipeline);

        Ok(())
    }
}

struct GeneratorPipeline {
    model: MixFormerSequentialForCausalLM,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl GeneratorPipeline {
    fn generate(&mut self, prompt: &str, max_length: usize) -> Result<(String, f64), Error> {
        // Like I understand anything here
        self.model.clear_kv_cache();

        let tokens = self.tokenizer.encode(prompt, true).unwrap();
        if tokens.is_empty() {
            bail!("Prompt is empty");
        }

        let mut generated_tokens = 0usize;
        let mut tokens = tokens.get_ids().to_vec();
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => bail!("Cannot find the endoftext token"),
        };

        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..max_length {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;

            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).unwrap();
            output.push_str(&token);
        }

        Ok((output, start_gen.elapsed().as_secs_f64()))
    }
}
