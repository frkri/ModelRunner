use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use sqlx::SqlitePool;

use crate::api::auth::Auth;
use crate::api::client::{ApiClient, Permission};

#[allow(dead_code)]
#[path = "../api/mod.rs"]
mod api;

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

        /// Creator ID
        #[clap(short, long)]
        creator_id: Option<String>,

        /// Scope of permission that the key will have
        #[clap(short, long, value_parser, num_args = 1.., value_delimiter = ',', default_values_t = vec ! [Permission::UseSelf, Permission::StatusSelf, Permission::DeleteSelf, Permission::UpdateSelf])]
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
        Commands::GenerateKey {
            name,
            permission,
            creator_id,
        } => {
            let client =
                ApiClient::new(&state.auth, &name, &permission, &creator_id, &state.db_pool)
                    .await?;
            println!("Generated new API client:\n{}", &client);
        }
    }
    Ok(())
}
