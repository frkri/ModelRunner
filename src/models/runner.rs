use std::io::Cursor;

use anyhow::{bail, Error, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops::softmax;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use candle_transformers::models::whisper::quantized_model::Whisper;
use candle_transformers::models::whisper::{
    audio, Config, COMPRESSION_RATIO_THRESHOLD, EOT_TOKEN, HOP_LENGTH, LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD, NO_SPEECH_TOKENS, NO_TIMESTAMPS_TOKEN, SAMPLE_RATE, SOT_TOKEN,
    TEMPERATURES, TRANSCRIBE_TOKEN, TRANSLATE_TOKEN,
};
use candle_transformers::models::{mixformer, whisper};
use candle_transformers::quantized_var_builder::VarBuilder;
use hf_hub::api::sync::ApiRepo;
use log::{debug, error};
use rand::distributions::Distribution;
use rand::random;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::models::pcm_decode::pcm_decode;

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/phi/main.rs
pub struct TextGeneratorPipeline {
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

impl TextGeneratorPipeline {
    pub(crate) fn clone(&self) -> TextGeneratorPipeline {
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
    ) -> Result<TextGeneratorPipeline, Error> {
        let tokenizer_file = repo.get(tokenizer_filename)?;
        let gguf_file = repo.get(gguf_filename)?;

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(gguf_file, &device)?;
        let model = MixFormerSequentialForCausalLM::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();

        let pipeline = TextGeneratorPipeline {
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
    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<(String, f64)> {
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

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/whisper/main.rs
pub struct AudioGeneratorPipeline {
    model: Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    timestamps: bool,
    seed: rand::rngs::StdRng,
}

impl AudioGeneratorPipeline {
    pub(crate) fn clone(&self) -> AudioGeneratorPipeline {
        AudioGeneratorPipeline {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            config: self.config.clone(),
            mel_filters: self.mel_filters.clone(),
            suppress_tokens: self.suppress_tokens.clone(),
            sot_token: self.sot_token,
            transcribe_token: self.transcribe_token,
            translate_token: self.translate_token,
            eot_token: self.eot_token,
            no_speech_token: self.no_speech_token,
            no_timestamps_token: self.no_timestamps_token,
            timestamps: self.timestamps,
            seed: self.seed.clone(),
        }
    }
}

impl AudioGeneratorPipeline {
    pub fn with_gguf_model(
        repo: ApiRepo,
        config_filename: &str,
        tokenizer_filename: &str,
        gguf_filename: &str,
        mel_filters_filename: &str,
        timestamps: bool,
        seed: rand::rngs::StdRng,
    ) -> Result<AudioGeneratorPipeline> {
        let config_path = repo.get(config_filename)?;
        let tokenizer_path = repo.get(tokenizer_filename)?;
        let model_path = repo.get(gguf_filename)?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let vb = VarBuilder::from_gguf(model_path, &Device::Cpu)?;
        let model = Whisper::load(&vb, config.clone())?;

        let mel_bytes = &*std::fs::read(mel_filters_filename)?;
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &Device::Cpu)?;
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => bail!("Unable to find any non-speech token"),
            Some(n) => n,
        };

        Ok(AudioGeneratorPipeline {
            model,
            tokenizer,
            config,
            mel_filters,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            timestamps,
            seed,
        })
    }

    pub fn transcribe(&mut self, input: Box<[u8]>, language_token: &str) -> Result<Vec<Segment>> {
        let mel = self.load_mel(input)?;
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        let language_token = match token_id(&self.tokenizer, &format!("<|{language_token}|>")) {
            Ok(token_id) => token_id,
            Err(_) => bail!("language {language_token} is not supported"),
        };

        while seek < content_frames {
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, whisper::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, language_token)?;
            seek += segment_size;
            if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                debug!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                debug!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self.tokenizer.decode(&tokens_to_decode, true).unwrap();
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self.tokenizer.decode(&tokens_to_decode, true).unwrap();
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                debug!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            segments.push(segment)
        }
        Ok(segments)
    }

    fn decode_with_fallback(
        &mut self,
        segment: &Tensor,
        language_token: u32,
    ) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t, language_token);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    error!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn decode(&mut self, mel: &Tensor, t: f64, language_token: u32) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(&mel, true)?;
        debug!("audio features: {:?}", audio_features.dims());

        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        tokens.push(language_token);
        tokens.push(self.transcribe_token);

        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.seed) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).unwrap();
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn load_mel(&self, input: Box<[u8]>) -> Result<Tensor> {
        let cursor = Cursor::new(input);
        let (pcm_data, sample_rate) = pcm_decode(cursor)?;
        if sample_rate != SAMPLE_RATE as u32 {
            bail!("Input file must have a {} sampling rate", SAMPLE_RATE)
        }
        debug!("pcm data loaded {}", pcm_data.len());
        let mel = audio::pcm_to_mel(&self.config, &pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &Device::Cpu,
        )?;
        debug!("loaded mel: {:?}", mel.dims());
        Ok(mel)
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}
