use rand::random;
use serde::Deserialize;

#[derive(Deserialize, Debug, Copy, Clone)]
pub struct GeneralModelConfig {
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_context_size: usize,
}

impl Default for GeneralModelConfig {
    #[tracing::instrument(level = "trace", skip())]
    fn default() -> Self {
        Self {
            seed: random(),
            temperature: Some(0.6),
            top_p: Some(0.6),
            repeat_penalty: 1.1,
            repeat_context_size: 64,
        }
    }
}
