use std::{fs::File, io::{self, Read}};

use clann::core::Config;
use indicatif::{ProgressBar, ProgressStyle};

pub mod db_utils;

pub fn load_configs_from_file(path: &str) -> io::Result<Vec<Config>> {
    let mut file = File::open(path)?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let configs: Vec<Config> = serde_json::from_str(&json)?;
    Ok(configs)
}


pub fn print_benchmark_header(name: &str) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ {:<61} ║", format!("{}", name));
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

pub fn create_progress_bar(name: String, total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Failed to set progress bar style")
        .progress_chars("=>-"));
    pb.set_message(name);
    pb
}