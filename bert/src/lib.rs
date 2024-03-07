use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use tokenizers::Tokenizer;

pub struct Args {
    /// Run on CPU rather than GPU.
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    pub tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    pub model_id: Option<String>,
    pub revision: Option<String>,

    /// When set, compute embedding for this prompt.
    pub prompt: String,

    /// Use the PyTorch weights rather than the safetensors ones
    pub use_pth: bool,

    /// The number of times to run the prompt.
    pub n: usize,

    /// L2 normalization for embeddings.
    pub normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    pub approximate_gelu: bool,
}

impl Args {
    pub fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
        let device = common::device(self.cpu)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let var_builder = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(var_builder, &config)?;

        Ok((model, tokenizer))
    }
}

pub fn run(args: Args) -> Result<Tensor> {
    env_logger::init();
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if args.tracing {
        info!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer
        .encode(args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    info!("Loaded and encoded {:?}", start.elapsed());

    let mut response = Tensor::zeros((1, 1), DTYPE, &device)?;

    for idx in 0..args.n {
        let start = std::time::Instant::now();
        let ys = model.forward(&token_ids, &token_type_ids)?;
        if idx == 0 {
            response = ys.clone();
        }
        info!("Took {:?}", start.elapsed());
    }

    Ok(response)
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
