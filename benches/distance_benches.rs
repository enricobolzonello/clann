/// This tests serves two purposes:
/// 1. Compare on the same datasets and configs the puffinn and clann implementation
/// 2. Comparing different configurations for clann, since results will be stored in the db
/// 
use clann::core::Config;
use clann::metricdata::{AngularData, MetricData};
use clann::puffinn_binds::puffinn_index::get_distance_computations;
use clann::puffinn_binds::PuffinnIndex;
use clann::utils::load_hdf5_dataset;
use clann::utils::metrics::MetricsGranularity;
use clann::{build, init_with_config, save_metrics, search};
use criterion::{criterion_group, criterion_main, Criterion};
use rusqlite::{params, Connection};

use std::time::{Duration, Instant};
use utils::db_utils::{
    check_configuration_exists_clann, check_configuration_exists_puffinn, BenchmarkError,
};
use utils::{
    create_progress_bar, load_configs_from_file, print_benchmark_header,
    DB_PATH,
};

mod utils;


fn run_benchmark_config_clann(
    config: &Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = format!("./datasets/{}.hdf5", config.dataset_name);
    let (data_raw, queries, ground_truth_distances) = load_hdf5_dataset(&dataset_path)?;

    let data = AngularData::new(data_raw);
    let n = data.num_points();

    // Attempt to build the clustered index, but skip if it fails
    let mut clustered_index = match init_with_config(data, config.clone()) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("Failed to initialize clustered index: {:?}", e);
            return Err(Box::new(e));
        }
    };

    // Skip build if it fails
    match build(&mut clustered_index) {
        Ok(_) => {
            clustered_index
                .enable_metrics()
                .expect("Failed to enable metrics");
        }
        Err(e) => {
            eprintln!("Failed to build clustered index: {:?}", e);
            return Err(Box::new(e));
        }
    }

    let mut clustered_counts = Vec::new();
    let mut distance_results = Vec::with_capacity(queries.nrows());

    let search_start = Instant::now();
    // run all queries (CLANN)
    for query in queries.rows() {
        let query_slice = query.as_slice().expect("Failed to get query slice");

        let result = search(&mut clustered_index, query_slice)?;
        distance_results.push(result);

        let clustered_count = clustered_index
            .metrics
            .as_ref()
            .expect("Metrics not enabled")
            .current_query()
            .unwrap()
            .distance_computations as u32;

        clustered_counts.push(clustered_count);
    }
    let total_search_time = search_start.elapsed();

    let distances: Vec<Vec<f32>> = distance_results
        .iter()
        .map(|result| result.iter().map(|&(distance, _)| distance).collect())
        .collect();

    save_metrics(
        &mut clustered_index,
        DB_PATH,
        MetricsGranularity::Query,
        &ground_truth_distances,
        &distances,
        n,
        &total_search_time,
    )?;

    Ok(())
}

fn run_benchmark_config_puffinn(
    config: &Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = format!("./datasets/{}.hdf5", config.dataset_name);
    let (data_raw, queries, _) = load_hdf5_dataset(&dataset_path)?;

    let data = AngularData::new(data_raw);
    let n = data.num_points();

    // create index
    let base_index = PuffinnIndex::new(&data, config.kb_per_point * data.num_points() * 1024)
        .expect("Failed to initialize PUFFINN index");

    let mut puffinn_counts = Vec::new();
    let mut query_times = Vec::new();
    // run all queries (PUFFINN)
    let search_start = Instant::now();
    for query in queries.rows() {
        let query_slice = query.as_slice().expect("Failed to get query slice");

        let query_query_start = Instant::now();
        base_index
            .search::<AngularData<ndarray::OwnedRepr<f32>>>(query_slice, config.k, config.delta)
            .expect("PUFFINN search failed");
        let query_time = query_query_start.elapsed();

        let puffinn_count = get_distance_computations();

        puffinn_counts.push(puffinn_count);
        query_times.push(query_time);
    }
    let total_search_time = search_start.elapsed();

    let conn = Connection::open(DB_PATH)?;
    save_puffinn_results(
        &conn,
        config,
        n,
        total_search_time,
        &puffinn_counts,
        &query_times,
    )?;

    Ok(())
}

fn save_puffinn_results(
    conn: &Connection,
    config: &Config,
    dataset_len: usize,
    total_search_time: Duration,
    puffinn_counts: &[u32],
    query_times: &[Duration],
) -> Result<(), rusqlite::Error> {
    let memory_used_bytes = config.kb_per_point * dataset_len * 1024;

    // Insert overall results
    conn.execute(
        "INSERT OR REPLACE INTO puffinn_results 
        (kb_per_point, k, delta, dataset, dataset_len, memory_used_bytes, 
         total_time_ms, queries_per_second) 
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            config.kb_per_point,
            config.k,
            config.delta,
            config.dataset_name,
            dataset_len,
            memory_used_bytes,
            total_search_time.as_millis() as i64,
            puffinn_counts.len() as f64 / total_search_time.as_secs_f64()
        ],
    )?;

    // Insert per-query results
    let mut stmt = conn.prepare(
        "INSERT INTO puffinn_results_query 
        (kb_per_point, k, delta, dataset, query_idx, query_time_ms, distance_computations) 
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;

    for (idx, (count, query_time)) in puffinn_counts.iter().zip(query_times.iter()).enumerate() {
        stmt.execute(params![
            config.kb_per_point,
            config.k,
            config.delta,
            config.dataset_name,
            idx,
            query_time.as_millis() as i64,
            *count
        ])?;
    }

    Ok(())
}

pub fn compare_implementations_distance() -> Result<(), Box<dyn std::error::Error>> {
    let configs = load_configs_from_file("benches/configs.json")?;

    let conn = Connection::open(DB_PATH)?;
    let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

    for (config_idx, config) in configs.iter().enumerate() {

        // run clann
        match check_configuration_exists_clann(&conn, config, git_hash) {
            Ok(false) => {
                // Run benchmark, catching any errors for this specific configuration
                match run_benchmark_config_clann(config) {
                    Ok(_) => {
                        println!("CLANN config {} run", config_idx);
                    }
                    Err(e) => {
                        println!(
                            "Error running benchmark for configuration {}: {}",
                            config_idx, e
                        );
                    }
                }
            }
            Ok(true) => {
                println!("Configuration {} already exists (unexpected)", config_idx);
            }
            Err(BenchmarkError::ConfigExists(msg)) => {
                println!("Skipping configuration {}: {}", config_idx, msg);
            }
            Err(e) => {
                return Err(Box::new(e));
            }
        }

        match check_configuration_exists_puffinn(&conn, config) {
            Ok(false) => {
                // run puffinn
                match run_benchmark_config_puffinn(config) {
                    Ok(_) => {
                        println!("PUFFINN config {} run", config_idx);
                    }
                    Err(e) => {
                        println!(
                            "Error running puffinn benchmark for configuration {}: {}",
                            config_idx, e
                        );
                    }
                }
            }
            Ok(true) => {
                println!("Configuration {} already exists", config_idx);
            }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
    }

    Ok(())
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
