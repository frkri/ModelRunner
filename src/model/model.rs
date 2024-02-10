use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ModelBase {
    /// The name of the model
    pub name: String,

    /// The license of the model
    pub license: String,

    /// The domain that the model is designed for including the tasks it can perform
    pub domain: ModelDomain,

    /// The id of the model repository
    pub repo_id: String,

    /// The revision of the model repository
    pub repo_revision: String,
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
