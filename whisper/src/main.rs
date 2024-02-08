use anyhow::Result;
use clap::Parser;
use whisper::{transcribe, Args};

fn main() -> Result<()> {
    let args = Args::parse();

    transcribe(args)?;

    Ok(())
}
