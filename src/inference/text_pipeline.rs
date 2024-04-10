use std::path::PathBuf;

use anyhow::{bail, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as Phi2;
use candle_transformers::models::quantized_stable_lm::Model as QStableLM;
use candle_transformers::models::stable_lm::Config as StableLmConfig;
use candle_transformers::quantized_var_builder::VarBuilder;
use hf_hub::api::sync::ApiRepo;
use rand::random;
use tokenizers::Tokenizer;

use crate::inference::token_output_stream::TokenOutputStream;

// Taken from
// https://github.com/huggingface/candle/blob/main/candle-examples/examples/phi/main.rs
// https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs
// https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
// https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-lm
pub struct TextGeneratorPipeline {
    pub model: Model,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_context_size: usize,
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Clone)]
pub enum Model {
    Phi(Option<Phi2>),
    Mistral(Option<ModelWeights>),
    OpenHermes(Option<ModelWeights>),
    StableLm(Option<QStableLM>),
}

pub enum ModelConfig {
    Phi(mixformer::Config),
    StableLm(StableLmConfig),
}

impl Clone for TextGeneratorPipeline {
    fn clone(&self) -> TextGeneratorPipeline {
        TextGeneratorPipeline {
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

impl TextGeneratorPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn with_quantized_gguf_config(
        repo: &ApiRepo,
        model: &Model,
        config: ModelConfig,
        tokenizer_filename: &str,
        gguf_filename: &str,
        seed: Option<u64>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_context_size: usize,
    ) -> Result<TextGeneratorPipeline> {
        let tokenizer_file = repo.get(tokenizer_filename)?;
        let gguf_file = repo.get(gguf_filename)?;

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(gguf_file, &device)?;
        let model = match model {
            Model::Phi(_) => {
                let ModelConfig::Phi(config) = config else {
                    bail!("Invalid model config")
                };
                let model = Phi2::new(&config, vb)?;
                Model::Phi(Some(model))
            }
            Model::StableLm(_) => {
                let ModelConfig::StableLm(config) = config else {
                    bail!("Invalid model config")
                };
                let model = QStableLM::new(&config, vb)?;
                Model::StableLm(Some(model))
            }
            _ => bail!("Unsupported model"),
        };
        let tokenizer = TokenOutputStream::new(Tokenizer::from_file(tokenizer_file).unwrap());

        let pipeline = TextGeneratorPipeline {
            model,
            device,
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

    #[allow(clippy::too_many_arguments)]
    pub fn with_quantized_gguf(
        repo: &ApiRepo,
        model: &Model,
        tokenizer_file: PathBuf,
        gguf_filename: &str,
        seed: Option<u64>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_context_size: usize,
    ) -> Result<TextGeneratorPipeline> {
        let gguf_file = repo.get(gguf_filename)?;
        let mut file = std::fs::File::open(&gguf_file)?;

        let device = Device::Cpu;
        let model_reader =
            gguf_file::Content::read(&mut file).map_err(|e| e.with_path(gguf_file))?;
        let model_weights = Some(ModelWeights::from_gguf(model_reader, &mut file, &device)?);
        let tokenizer = TokenOutputStream::new(Tokenizer::from_file(tokenizer_file).unwrap());

        let pipeline = TextGeneratorPipeline {
            model: match model {
                Model::Mistral(_) => Model::Mistral(model_weights),
                Model::OpenHermes(_) => Model::OpenHermes(model_weights),
                _ => bail!("Unsupported model"),
            },
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
    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<(String, f64)> {
        if let Model::Phi(Some(ref mut m)) = self.model {
            m.clear_kv_cache();
        }
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            bail!("Prompt is empty");
        }

        let eos_token = match self.model {
            Model::Mistral(_) => match self.tokenizer.tokenizer().get_vocab(true).get("</s>") {
                Some(token) => *token,
                None => bail!("Cannot find </s> token"),
            },
            Model::OpenHermes(_) => 32000,
            _ => match self
                .tokenizer
                .tokenizer()
                .get_vocab(true)
                .get("<|endoftext|>")
            {
                Some(token) => *token,
                None => bail!("Cannot find the <|endoftext|> token"),
            },
        };

        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..max_length {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Phi(Some(model)) => model.forward(&input)?,
                Model::Mistral(Some(model)) => model.forward(&input, start_pos)?,
                Model::OpenHermes(Some(model)) => model.forward(&input, start_pos)?,
                Model::StableLm(Some(model)) => model.forward(&input, start_pos)?,
                _ => bail!("Model not initialized"),
            };
            let logits = match self.model {
                Model::Phi(_) => logits.squeeze(0)?.to_dtype(DType::F32)?,
                Model::Mistral(_) => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?,
                Model::OpenHermes(_) => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?,
                Model::StableLm(_) => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?,
            };
            let logits = if (self.repeat_penalty - 1.).abs() < f32::EPSILON {
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

            match self.tokenizer.next_token(next_token) {
                Ok(text) => {
                    if let Some(text) = text {
                        output.push_str(&text);
                    }
                }
                Err(err) => bail!("Cannot decode tokens: {err}"),
            };
        }
        match self.tokenizer.decode_rest() {
            Ok(text) => {
                if let Some(text) = text {
                    output.push_str(&text);
                }
            }
            Err(err) => bail!("Cannot decode tokens: {err}"),
        };

        Ok((output, start_gen.elapsed().as_secs_f64()))
    }
}
