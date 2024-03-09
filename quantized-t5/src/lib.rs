use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::quantized_t5 as t5};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use rand::{thread_rng, Rng};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

pub struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    pub tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    pub model_id: Option<String>,
    pub revision: Option<String>,

    pub weight_file: Option<String>,

    pub config_file: Option<String>,

    /// Enable/disable decoding.
    pub disable_cache: bool,

    /// Use this prompt
    pub prompt: String,

    /// The temperature used to generate samples.
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for repeat penalty.
    pub repeat_last_n: usize,

    /// The model size to use.
    pub which: Which,
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(T5ModelBuilder, Tokenizer)> {
        let device = Device::Cpu;
        let default_model = "lmz/candle-quantized-t5".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, "main".to_string()),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &args.config_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("config.json")?,
                Which::FlanT5Small => api.get("config-flan-t5-small.json")?,
                Which::FlanT5Base => api.get("config-flan-t5-base.json")?,
                Which::FlanT5Large => api.get("config-flan-t5-large.json")?,
                Which::FlanT5Xl => api.get("config-flan-t5-xl.json")?,
                Which::FlanT5Xxl => api.get("config-flan-t5-xxl.json")?,
            },
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &args.weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("model.gguf")?,
                Which::FlanT5Small => api.get("model-flan-t5-small.gguf")?,
                Which::FlanT5Base => api.get("model-flan-t5-base.gguf")?,
                Which::FlanT5Large => api.get("model-flan-t5-large.gguf")?,
                Which::FlanT5Xl => api.get("model-flan-t5-xl.gguf")?,
                Which::FlanT5Xxl => api.get("model-flan-t5-xxl.gguf")?,
            },
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let var_builder = t5::VarBuilder::from_gguf(&self.weights_filename, &self.device)?;
        Ok(t5::T5ForConditionalGeneration::load(
            var_builder,
            &self.config,
        )?)
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}

pub fn encode(args: &Args) -> Result<Tensor> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let device = &builder.device;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let input_tokens_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let mut model = builder.build_model()?;
    let ys = model.encode(&input_tokens_ids)?;
    Ok(ys)
}

pub fn conditional_generation(args: &Args) -> Result<String> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let device = &builder.device;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let input_tokens_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let mut model = builder.build_model()?;
    let mut output_tokens_ids = [builder
        .config
        .decoder_start_token_id
        .unwrap_or(builder.config.pad_token_id) as u32]
    .to_vec();

    let temperature = if args.temperature <= 0. {
        None
    } else {
        Some(args.temperature)
    };

    let mut rng = thread_rng();
    let mut logits_processor = LogitsProcessor::new(rng.gen(), temperature, args.top_p);
    let encoder_output = model.encode(&input_tokens_ids)?;
    let mut result = String::new();
    for index in 0.. {
        if output_tokens_ids.len() > 512 {
            break;
        }
        let decoder_token_ids = if index == 0 || !builder.config.use_cache {
            Tensor::new(output_tokens_ids.as_slice(), device)?.unsqueeze(0)?
        } else {
            let last_token = *output_tokens_ids.last().unwrap();
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };
        let logits = model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_tokens_ids.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &output_tokens_ids[start_at..],
            )?
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == builder.config.eos_token_id {
            break;
        }
        output_tokens_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            result = format!("{}{}", result, text);
        }
    }
    Ok(result)
}
