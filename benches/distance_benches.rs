/// This tests serves two purposes:
/// 1. Compare on the same datasets and configs the puffinn and clann implementation
/// 2. Comparing different configurations for clann, since results will be stored in the db

use std::path::Path;
use std::time::{Duration, Instant};
use clann::core::Config;
use clann::puffinn_binds::puffinn_index::get_distance_computations;
use clann::{build, init_with_config, save_metrics, search};
use clann::metricdata::{AngularData, MetricData};
use clann::puffinn_binds::PuffinnIndex;
use clann::utils::load_hdf5_dataset;
use clann::utils::metrics::MetricsGranularity;
use criterion::{criterion_group, criterion_main, Criterion};
use rusqlite::Connection;
use serde::Serialize;
use utils::db_utils::{check_configuration_exists, BenchmarkError};
use utils::{create_progress_bar, load_configs_from_file, print_benchmark_header, DB_PATH};

mod utils;

#[derive(Serialize)]
struct DistanceMetric {
    config_id: usize,
    query_id: usize,
    method: String,
    value: u32,
    dataset: String,
    k: usize,
    delta: f32,
}

#[derive(Default)]
struct BenchmarkResults {
    clustered_counts: Vec<Vec<u32>>,
    puffinn_counts: Vec<Vec<u32>>,
    search_times: Vec<Duration>,
    distance_results: Vec<Vec<Vec<f32>>>,
}

impl BenchmarkResults {
    fn new() -> Self {
        Self::default()
    }

    fn add_config_results(
        &mut self,
        clustered: Vec<u32>,
        puffinn: Vec<u32>,
        search_time: Duration,
        distances: Vec<Vec<f32>>,
    ) {
        self.clustered_counts.push(clustered);
        self.puffinn_counts.push(puffinn);
        self.search_times.push(search_time);
        self.distance_results.push(distances);
    }

    fn export_to_csv<P: AsRef<Path>>(&self, configs: &[Config], output_path: P) -> Result<(), Box<dyn std::error::Error>> {
        let mut metrics = Vec::new();

        for (config_id, config) in configs.iter().enumerate() {
            // Add clustered metrics
            for (query_id, &value) in self.clustered_counts[config_id].iter().enumerate() {
                metrics.push(DistanceMetric {
                    config_id,
                    query_id,
                    method: "clustered".to_string(),
                    value,
                    dataset: config.dataset_name.clone(),
                    k: config.k,
                    delta: config.delta,
                });
            }

            // Add PUFFINN metrics
            for (query_id, &value) in self.puffinn_counts[config_id].iter().enumerate() {
                metrics.push(DistanceMetric {
                    config_id,
                    query_id,
                    method: "puffinn".to_string(),
                    value,
                    dataset: config.dataset_name.clone(),
                    k: config.k,
                    delta: config.delta,
                });
            }
        }

        let mut writer = csv::Writer::from_path(output_path)?;
        for metric in metrics {
            writer.serialize(metric)?;
        }
        writer.flush()?;

        Ok(())
    }

    fn print_summary(&self, config_idx: usize) {
        let clustered = &self.clustered_counts[config_idx];
        let puffinn = &self.puffinn_counts[config_idx];

        let clustered_avg = clustered.iter().sum::<u32>() as f64 / clustered.len() as f64;
        let puffinn_avg = puffinn.iter().sum::<u32>() as f64 / puffinn.len() as f64;

        let clustered_std = compute_std(clustered, clustered_avg);
        let improvement = compute_improvement(clustered_avg, puffinn_avg);

        let (min_c, max_c) = get_min_max(clustered);
        let (min_p, max_p) = get_min_max(puffinn);

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<15.2} {:<15.2}% {:<20} {:<20}",
            config_idx,
            clustered_avg,
            puffinn_avg,
            clustered_std,
            improvement,
            format!("{}/{}", min_c, min_p),
            format!("{}/{}", max_c, max_p)
        );
    }
}

fn compute_std(values: &[u32], mean: f64) -> f64 {
    (values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn compute_improvement(clustered_avg: f64, puffinn_avg: f64) -> f64 {
    ((puffinn_avg - clustered_avg) / puffinn_avg * 100.0).max(-999.9)
}

fn get_min_max(values: &[u32]) -> (u32, u32) {
    (
        *values.iter().min().unwrap_or(&0),
        *values.iter().max().unwrap_or(&0),
    )
}

pub fn compare_implementations_distance() -> Result<(), Box<dyn std::error::Error>> {
    let configs = load_configs_from_file("benches/configs.json")?;
    let mut results = BenchmarkResults::new();

    let conn = Connection::open(DB_PATH)?;
    let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

    println!("\nDistance Computation Comparison Summary:");
    println!("{:-<120}", "");
    println!("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<20}", 
        "Config", "Clustered Avg", "PUFFINN Avg", "Clustered Std", "Improvement %", 
        "Min (C/P)", "Max (C/P)");
    println!("{:-<120}", "");

    for (config_idx, config) in configs.iter().enumerate() {

        match check_configuration_exists(&conn, config, git_hash) {
            Ok(false) => {
                // Configuration doesn't exist, run the benchmark
                let (clustered_counts, puffinn_counts, search_time, distance_results) = 
                    run_benchmark_config(config)?;

                results.add_config_results(
                    clustered_counts,
                    puffinn_counts,
                    search_time,
                    distance_results,
                );

                results.print_summary(config_idx);
            }
            Ok(true) => {
                // This shouldn't happen due to our error handling
                println!("Configuration {} already exists (unexpected)", config_idx);
                continue;
            }
            Err(BenchmarkError::ConfigExists(msg)) => {
                results.add_config_results(vec![], vec![], Duration::default(), vec![]);
                println!("Skipping configuration {}: {}", config_idx, msg);
                continue;
            }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
    }

    results.export_to_csv(&configs, "./runs/distance_computations.csv")?;

    Ok(())
}

fn run_benchmark_config(
    config: &Config,
) -> Result<(Vec<u32>, Vec<u32>, Duration, Vec<Vec<f32>>), Box<dyn std::error::Error>> {
    let dataset_path = format!("./datasets/{}.hdf5", config.dataset_name);
    let (data_raw, queries, ground_truth_distances) = load_hdf5_dataset(&dataset_path)?;

    let data = AngularData::new(data_raw);
    let n = data.num_points();

    // create the two indexes
    let base_index = PuffinnIndex::new(&data, config.kb_per_point * data.num_points() * 1024).expect("Failed to initialize PUFFINN index");
    let mut clustered_index = init_with_config(data, config.clone()).expect("Failed to initialize clustered index");
    clustered_index.enable_metrics().expect("Failed to enable metrics");
    build(&mut clustered_index).expect("Failed to build clustered index");
    
    let mut clustered_counts = Vec::new();
    let mut puffinn_counts = Vec::new();
    let mut distance_results = Vec::with_capacity(queries.nrows());
    
    let search_start = Instant::now();
    // run all queries
    for query in queries.rows() {
        let query_slice = query.as_slice().expect("Failed to get query slice");

        let result = search(&mut clustered_index, query_slice).unwrap();

        let distances: Vec<f32> = result.iter()
            .map(|&(distance, _)| distance)
            .collect();
        distance_results.push(distances);

        let clustered_count = clustered_index
            .metrics
            .as_ref()
            .expect("Metrics not enabled")
            .current_query()
            .unwrap().distance_computations as u32;

        base_index
            .search::<AngularData<ndarray::OwnedRepr<f32>>>(query_slice, config.k, config.delta)
            .expect("PUFFINN search failed");
        let puffinn_count = get_distance_computations();

        clustered_counts.push(clustered_count);
        puffinn_counts.push(puffinn_count);
    }
    let total_search_time = search_start.elapsed();

    save_metrics(
        &mut clustered_index,
        DB_PATH,
        MetricsGranularity::Query,
        &ground_truth_distances,
        &distance_results,
        n,
        &total_search_time,
    )?;

    Ok((clustered_counts, puffinn_counts, total_search_time, distance_results))
}

pub fn run_distance_benchmarks(_c: &mut Criterion) {
    print_benchmark_header("PUFFINN-CLANN Distance Computations Comparison");
    let pb = create_progress_bar("Running distance comparison".to_string(), 100);
    compare_implementations_distance().expect("Error in compare implem");
    pb.finish_with_message("Distance Comparison complete");
}

criterion_group! {
    name = distance_benches;
    config = Criterion::default().configure_from_args();
    targets = run_distance_benchmarks
}

criterion_main!(distance_benches);