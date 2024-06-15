use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelBase {
    /// The name of the models
    pub name: String,

    /// The license of the models
    pub license: String,

    /// The domain that the models is designed for including the tasks it can perform
    pub domain: ModelDomain,

    /// The id of the models repository
    pub repo_id: String,

    /// The revision of the models repository
    pub repo_revision: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ModelDomain {
    Text(Vec<TextTask>),
    Video(Vec<VideoTask>),
    Audio(AudioTask),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum TextTask {
    Chat,
    Extract,
    Instruct,
    Sentiment,
    Translate,
    Identify,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum VideoTask {
    Describe,
    Generate,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AudioTask {
    Transcribe,
}
