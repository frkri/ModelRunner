use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use sqlx::SqlitePool;

use crate::api::api_client::Permission;
use crate::api::auth::Auth;

#[allow(dead_code)]
#[path = "../api/mod.rs"]
mod api;

// TODO: This won't be feature complete for some time
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The SQLite database file path
    #[arg(short, long, env, default_value = "model_runner.db")]
    pub sqlite_file_path: String,

    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    GenerateKey {
        /// Name of the API key
        #[clap(short, long)]
        name: String,

        /// Scope of permission that the key will have
        #[clap(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
        permission: Vec<Permission>,
    },
}

struct AppState {
    db_pool: SqlitePool,
    auth: Auth,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let db_pool = SqlitePool::connect(&args.sqlite_file_path).await?;
    let auth = Auth::default();
    let state = AppState { db_pool, auth };

    match args.cmd {
        Commands::GenerateKey { name, permission } => {
            let key = state
                .auth
                .create_api_key(name.as_str(), &permission, &state.db_pool)
                .await?;
            println!(
                "Generated new API key with {:#?} permissions\n{}",
                &permission, &key
            );
        }
    }
    Ok(())
}
