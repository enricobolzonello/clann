use criterion::{criterion_group, criterion_main, Criterion};
use distance::compare_implementations_distance;
use indicatif::{ProgressBar, ProgressStyle};
use time::compare_implementations_time;

mod time;
mod distance;
mod utils;

const DATASET_PATH: &str = "/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5";

fn print_benchmark_header(name: &str) {
    println!("\n{}", "╔═══════════════════════════════════════════════════════════════╗");
    println!("║ {:<61} ║", format!("{}", name));
    println!("{}", "╚═══════════════════════════════════════════════════════════════╝");
}

fn create_progress_bar(name: String, total: u64) -> ProgressBar {
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

pub fn run_time_benchmarks(c: &mut Criterion) {
    print_benchmark_header("PUFFINN-CLANN Time Comparison");
    let pb = create_progress_bar("Running time comparison".to_string(), 100);
    compare_implementations_time(c, DATASET_PATH);
    pb.finish_with_message("Time Comparison complete");
}

pub fn run_distance_benchmarks(c: &mut Criterion) {
    print_benchmark_header("PUFFINN-CLANN Distance Computations Comparison");
    let pb = create_progress_bar("Running distance comparison".to_string(), 100);
    compare_implementations_distance(c, DATASET_PATH);
    pb.finish_with_message("Distance Comparison complete");
}

criterion_group! {
    name = distance_benches;
    config = Criterion::default()
        .configure_from_args();
    targets = run_distance_benchmarks
}

criterion_group! {
    name = time_benches;
    config = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = run_time_benchmarks
}

criterion_main!(distance_benches, time_benches);