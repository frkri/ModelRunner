use anyhow::{Context, Result};
use axum::Router;
use axum::routing::get;
use clap::Parser;
use clap_serde_derive::ClapSerde;
use log::{error, info};
use tokio::net::TcpListener;

use crate::config::Config;

mod config;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, default_value = "ModelRunner.toml")]
    config_path: String,

    /// Configuration options
    #[command(flatten)]
    pub opt_config: <Config as ClapSerde>::Opt,
}


#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let args = Args::parse();
    let config = Config::from_toml(&args.config_path).context(format!("Failed to load config from file {}", &args.config_path))?.merge(args.opt_config);

    let router = Router::new().route("/", get(|| async { "Hello, World!" }));

    let listener = TcpListener::bind(format!("{}:{}", config.address, config.port)).await?;
    info!("Listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, router).with_graceful_shutdown(shutdown_signal()).await?;
    Ok(())
}

async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => info!("Shutting down..."),
        Err(e) => error!("Failed to listen for shutdown signal: {}", e),
    }
}
