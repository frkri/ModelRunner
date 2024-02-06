use anyhow::Error;

use crate::model::task::instruct::{InstructHandler, InstructRequest, InstructResponse};

pub struct Phi2Model {
    pub name: String,
    pub download_url: String,
}
impl InstructHandler for Phi2Model {
    fn run(&self, params: InstructRequest) -> Result<InstructResponse, Error> {
        println!("Phi2Model with InstructRequest: {:?}", params);

        Ok(InstructResponse {
            output: "Phi2Model".to_string(),
        })
    }
}
