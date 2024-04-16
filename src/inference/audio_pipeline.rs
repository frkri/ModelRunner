#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::io::Cursor;

use anyhow::{bail, Result};
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper;
use candle_transformers::models::whisper::quantized_model::Whisper;
use candle_transformers::models::whisper::{
    audio, Config, COMPRESSION_RATIO_THRESHOLD, EOT_TOKEN, HOP_LENGTH, LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD, NO_SPEECH_TOKENS, NO_TIMESTAMPS_TOKEN, SAMPLE_RATE, SOT_TOKEN,
    TEMPERATURES, TRANSCRIBE_TOKEN, TRANSLATE_TOKEN,
};
use candle_transformers::quantized_var_builder::VarBuilder;
use hf_hub::api::sync::ApiRepo;
use log::{debug, error};
use rand::distributions::Distribution;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::inference::pcm_decode::pcm_decode;

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

impl Clone for AudioGeneratorPipeline {
    fn clone(&self) -> Self {
        Self {
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
        repo: &ApiRepo,
        config_filename: &str,
        tokenizer_filename: &str,
        gguf_filename: &str,
        mel_filters_filename: &str,
        timestamps: bool,
        seed: rand::rngs::StdRng,
    ) -> Result<Self> {
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
        let start_of_transcript_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, TRANSLATE_TOKEN)?;
        let end_of_text_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => bail!("Unable to find any non-speech token"),
            Some(n) => n,
        };

        Ok(Self {
            model,
            tokenizer,
            config,
            mel_filters,
            suppress_tokens,
            sot_token: start_of_transcript_token,
            transcribe_token,
            translate_token,
            eot_token: end_of_text_token,
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
        let Ok(language_token) = token_id(&self.tokenizer, &format!("<|{language_token}|>")) else {
            bail!("language {language_token} is not supported")
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
            segments.push(segment);
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
                    error!("Error running at {t}: {err}");
                }
            }
        }
        unreachable!()
    }

    fn decode(&mut self, mel: &Tensor, t: f64, language_token: u32) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;
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
                no_speech_prob = f64::from(
                    softmax(&logits, 0)?
                        .i(self.no_speech_token as usize)?
                        .to_scalar::<f32>()?,
                );
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
                u32::try_from(distr.sample(&mut self.seed))?
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
            let prob = f64::from(
                softmax(&logits, D::Minus1)?
                    .i(next_token as usize)?
                    .to_scalar::<f32>()?,
            );
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).unwrap();
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
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
        if sample_rate != u32::try_from(SAMPLE_RATE)? {
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
