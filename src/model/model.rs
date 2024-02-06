use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Model {
    /// The name of the model
    name: String,

    /// The license of the model
    license: String,

    /// The domain that the model is designed for including the tasks it can perform
    domain: ModelDomain,

    /// The URL to the model file
    download_urls: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum ModelDomain {
    Text(Vec<TextTask>),
    Video(Vec<VideoTask>),
    Audio(AudioTask),
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum TextTask {
    Chat,
    Extract,
    Instruct,
    Sentiment,
    Translate,
    Identify,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum VideoTask {
    Describe,
    Generate,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum AudioTask {
    Transcribe,
}
