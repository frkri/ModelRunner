use std::env;
use std::error;
use reqwest::{Url};

fn main() -> Result<(), Box<dyn error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Missing URL argument")
    }

    let url = Url::parse(&args[1])?;

    let body = reqwest::blocking::get(url)?;
    if !body.status().is_success() {
       panic!("Request Failed!")
    }

    Ok(())
}
