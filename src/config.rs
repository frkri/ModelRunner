use clap_serde_derive::ClapSerde;
use serde::Deserialize;
use anyhow::Result;

#[derive(ClapSerde, Deserialize)]
pub struct Config {
    /// The address the listener binds to
    #[arg(short, long)]
    pub(crate) address: String,

    /// The port the listener binds to
    #[arg(short, long)]
    pub(crate) port: u16,
}

impl Config {
    pub fn from_toml(path: &str) -> Result<Self> {
        let str = std::fs::read_to_string(path)?;
        let config = toml::from_str(&str)?;
        Ok(config)
    }
}