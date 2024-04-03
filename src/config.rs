use anyhow::Result;
use clap::ArgAction;
use clap_serde_derive::ClapSerde;
use serde::Deserialize;

#[derive(ClapSerde, Deserialize)]
pub struct Config {
    /// The address the listener binds to
    #[arg(short, long, env, default_value = "0.0.0.0")]
    pub address: String,

    /// The port the listener binds to
    #[arg(short, long, env, default_value = "25566")]
    pub port: u16,

    /// The OpenTelemetry collector endpoint, enables telemetry
    #[arg(short, long, env)]
    pub otel_endpoint: Option<String>,

    /// Should the console output always be enabled even if the logs are pushed to a collector
    #[arg(long, env, action(ArgAction::SetTrue))]
    pub console: bool,

    /// Enable saving traces locally with tracing-chrome crate.
    /// This will save the traces in the current working directory as `trace-timestamp.json`
    #[arg(long, env, action(ArgAction::SetTrue))]
    pub trace_local: bool,

    /// The TLS configuration
    #[serde(default)]
    #[command(flatten)]
    pub tls: <TlsConfig as ClapSerde>::Opt,

    /// The SQLite database file path
    #[arg(short, long, env, default_value = "model_runner.db")]
    pub sqlite_file_path: String,
}

#[derive(ClapSerde, Deserialize, Debug)]
#[group(multiple = true)]
pub struct TlsConfig {
    /// The path to the certificate file in pem format. Must be used in conjunction with `private_key` option to enable TLS support otherwise it will error out
    #[arg(long, env, requires = "private_key")]
    pub certificate: String,

    /// The path to the private key file in pem format. Must be used in conjunction with `certificate` option to enable TLS support otherwise it will error out
    #[serde(alias = "private-key")]
    #[arg(long, env, requires = "certificate")]
    pub private_key: String,
}

impl Config {
    pub fn from_toml(path: &str) -> Result<Self> {
        let str = std::fs::read_to_string(path)?;
        let config = toml::from_str(&str)?;
        Ok(config)
    }
}
