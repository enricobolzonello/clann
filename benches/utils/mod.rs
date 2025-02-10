use std::{fs::File, io::{self, Read}};

use clann::core::Config;
use indicatif::{ProgressBar, ProgressStyle};

pub const DB_PATH: &str = "./clann_results.sqlite3";

pub mod db_utils;

pub fn load_configs_from_file(path: &str) -> io::Result<Vec<Config>> {
    let mut file = File::open(path)?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let configs: Vec<Config> = serde_json::from_str(&json)?;
    Ok(configs)
}


pub fn print_benchmark_header(name: &str) {
    println!("\n{}", "╔═══════════════════════════════════════════════════════════════╗");
    println!("║ {:<61} ║", format!("{}", name));
    println!("{}", "╚═══════════════════════════════════════════════════════════════╝");
}

pub fn create_progress_bar(name: String, total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .expect("Failed to create progress bar template")
            .progress_chars("#>-")
    );
    pb.set_message(name);
    pb
}