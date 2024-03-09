use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::t5};
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use rand::{thread_rng, Rng};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

pub struct Args {
    /// Run on CPU rather than GPU
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file)
    pub tracing: bool,

    /// The model repository to use on the HuggingFace Hub
    pub model_id: Option<String>,

    revision: Option<String>,

    /// Enable/disable decoding
    pub disable_cache: bool,

    /// Use this prompt
    pub prompt: String,

    /// If set along with --decode, will use this prompt to initialize the decoder.
    pub decoder_prompt: Option<String>,

    /// The temperature used to generate samples.
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for repeat penalty.
    pub repeat_last_n: usize,
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(T5ModelBuilder, Tokenizer)> {
        let device = common::device(args.cpu)?;
        let default_model = "flan-t5-base".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2"
        {
            common::hub_load_safetensors(&api, "model.safetensors.index.json")?
        } else {
            vec![api.get("model.safetensors")?]
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            T5ModelBuilder {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(var_builder, &self.config)?)
    }

    pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5ForConditionalGeneration::load(
            var_builder,
            &self.config,
        )?)
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

    let (builder, mut tokenizer) = T5ModelBuilder::load(args)?;
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
    let mut model = builder.build_encoder()?;
    let ys = model.forward(&input_tokens_ids)?;
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

    let (builder, mut tokenizer) = T5ModelBuilder::load(args)?;
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
    let mut model = builder.build_conditional_generation()?;
    let mut output_token_ids = [builder
        .config
        .decoder_start_token_id
        .unwrap_or(builder.config.pad_token_id) as u32]
    .to_vec();
    if let Some(decoder_prompt) = &args.decoder_prompt {
        info!("{decoder_prompt}");
        output_token_ids.extend(
            tokenizer
                .encode(decoder_prompt.to_string(), false)
                .map_err(E::msg)?
                .get_ids()
                .to_vec(),
        );
    }
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
        if output_token_ids.len() > 1024 {
            break;
        }
        let decoder_token_ids = if index == 0 || !builder.config.use_cache {
            Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };
        let logits = model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &output_token_ids[start_at..],
            )?
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == builder.config.eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            result = format!("{}{}", result, text);
        }
    }
    Ok(result)
}
