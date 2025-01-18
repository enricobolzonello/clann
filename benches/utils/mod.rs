use clann::core::Config;
use indicatif::{ProgressBar, ProgressStyle};

pub const CONFIGS: &[Config] = &[
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024, // 1GB
        num_clusters: 4,
        k: 10,
        delta: 0.9,
    },
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024,
        num_clusters: 8,
        k: 10,
        delta: 0.9,
    },
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024, 
        num_clusters: 12,
        k: 10,
        delta: 0.9,
    },
];

pub const DATASET_PATH: &str = "/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5";

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