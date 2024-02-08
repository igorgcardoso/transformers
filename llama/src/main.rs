use clap::Parser;
use llama::{generate, Args};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    generate(args)?;

    Ok(())
}
