use anyhow::{bail, Error};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use candle_transformers::quantized_var_builder::VarBuilder;
use hf_hub::api::sync::ApiRepo;
use rand::random;
use tokenizers::Tokenizer;

pub struct GeneratorPipeline {
    pub model: MixFormerSequentialForCausalLM,
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_context_size: usize,
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

impl GeneratorPipeline {
    pub(crate) fn clone(&self) -> GeneratorPipeline {
        GeneratorPipeline {
            model: self.model.clone(),
            device: self.device.clone(),
            tokenizer: self.tokenizer.clone(),
            logits_processor: LogitsProcessor::new(
                self.seed.unwrap_or(random()),
                self.temperature,
                self.top_p,
            ),
            repeat_penalty: self.repeat_penalty,
            repeat_context_size: self.repeat_context_size,
            seed: self.seed,
            temperature: self.temperature,
            top_p: self.top_p,
        }
    }
}

impl GeneratorPipeline {
    pub fn with_gguf_mixformer_model(
        repo: ApiRepo,
        config: mixformer::Config,
        tokenizer_filename: &str,
        gguf_filename: &str,
        seed: Option<u64>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_context_size: usize,
    ) -> Result<GeneratorPipeline, Error> {
        let tokenizer_file = repo.get(tokenizer_filename)?;
        let gguf_file = repo.get(gguf_filename)?;

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(gguf_file, &device)?;
        let model = MixFormerSequentialForCausalLM::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();

        let pipeline = GeneratorPipeline {
            model,
            device: Device::Cpu,
            tokenizer,
            logits_processor: LogitsProcessor::new(seed.unwrap_or(random()), temperature, top_p),
            repeat_penalty,
            repeat_context_size,
            seed,
            temperature,
            top_p,
        };

        Ok(pipeline)
    }
    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<(String, f64), Error> {
        self.model.clear_kv_cache();

        let tokens = self.tokenizer.encode(prompt, true).unwrap();
        if tokens.is_empty() {
            bail!("Prompt is empty");
        }

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
                let start_at = tokens.len().saturating_sub(self.repeat_context_size);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).unwrap();
            output.push_str(&token);
        }

        Ok((output, start_gen.elapsed().as_secs_f64()))
    }
}
