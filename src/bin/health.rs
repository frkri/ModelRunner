use reqwest::Url;
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Missing URL argument")
    }
    let url = Url::parse(&args[1]).unwrap();

    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .build()
        .unwrap();

    if !client.get(url).send().unwrap().status().is_success() {
        return ExitCode::from(1);
    }

    ExitCode::from(0)
}
