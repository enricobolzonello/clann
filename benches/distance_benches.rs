/// This tests serves two purposes:
/// 1. Compare on the same datasets and configs the puffinn and clann implementation
/// 2. Comparing different configurations for clann, since results will be stored in the db
///
    use clann::core::{Config, MetricsGranularity};
    use clann::metricdata::{AngularData, MetricData};
    use clann::puffinn_binds::puffinn::{get_distance_computations,PuffinnIndex};
    use clann::utils::load_hdf5_dataset;
    use clann::{build, init_from_file, init_with_config, save_metrics, search, serialize};
    use criterion::{criterion_group, criterion_main, Criterion};
    use env_logger::Env;
    use log::{error, info, warn};
    use ndarray::{Array, Ix2, OwnedRepr};
    use rusqlite::{params, Connection};

    use core::f32;
    use std::fs;
    use std::time::{Duration, Instant};
    use utils::db_utils::{
        check_configuration_exists_clann, check_configuration_exists_puffinn, BenchmarkError,
    };
    use utils::{create_progress_bar, load_configs_from_file, print_benchmark_header};

    mod utils;

    const INDEX_DIR: &str = "./__index_cache__";
    const DB_PATH: &str = "./results_v2.sqlite3";

    fn run_benchmark_config_clann(
        config: &Config,
        data: AngularData<OwnedRepr<f32>>,
        queries: &Array<f32, Ix2>,
        ground_truth_distances: &Array<f32, Ix2>,
        config_idx: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let index_path = format!(
            "{}/index_{}_k{:.2}_L{}.h5",
            INDEX_DIR, config.dataset_name, config.num_clusters_factor, config.num_tables
        );

        let mut clustered_index = if fs::metadata(&index_path).is_ok() {
            info!("Loading index from file: {}", index_path);
            init_from_file(data, &index_path).unwrap()
        } else {
            info!("No saved index found, initializing a new one");
            let mut new_index = init_with_config(data, config.clone()).unwrap();
            build(&mut new_index)
                .map_err(|e| eprintln!("Error: {}", e))
                .unwrap();
            serialize(&new_index, INDEX_DIR).unwrap();
            new_index
        };

        let mut clustered_counts = Vec::new();
        let mut distance_results = Vec::with_capacity(queries.nrows());

        let search_start = Instant::now();
        // run all queries (CLANN)
        let progress_bar = create_progress_bar(
            format!("CLANN config {}", config_idx),
            queries.nrows() as u64,
        );
        for query in queries.rows() {
            let query_slice = query.as_slice().expect("Failed to get query slice");

            let result = search(&mut clustered_index, query_slice)?;
            distance_results.push(result);

            let clustered_count = clustered_index.get_distance_computations()?;

            clustered_counts.push(clustered_count);
            progress_bar.inc(1);
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
            ground_truth_distances,
            &distances,
            &total_search_time,
        )?;

        Ok(())
    }

    fn run_benchmark_config_puffinn(
        config: &Config,
        data: &AngularData<OwnedRepr<f32>>,
        queries: &Array<f32, Ix2>,
        config_idx: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = data.num_points();

        info!("Creating PUFFINN index");
        // create index
        let (base_index, memory) =
            PuffinnIndex::new(data, config.num_tables).expect("Failed to initialize PUFFINN index");
        info!("PUFFINN index created with memory {}", memory);

        let mut puffinn_counts = Vec::new();
        let mut query_times = Vec::new();
        // run all queries (PUFFINN)
        info!("Starting search");
        let search_start = Instant::now();

        let progress_bar = create_progress_bar(
            format!("PUFFINN config {}", config_idx),
            queries.nrows() as u64,
        );
        for query in queries.rows() {
            let query_slice = query.as_slice().expect("Failed to get query slice");

            let query_query_start = Instant::now();
            base_index
                .search::<AngularData<ndarray::OwnedRepr<f32>>>(
                    query_slice,
                    config.k,
                    f32::INFINITY,
                    config.delta,
                )
                .expect("PUFFINN search failed");
            let query_time = query_query_start.elapsed();

            let puffinn_count = get_distance_computations();

            puffinn_counts.push(puffinn_count);
            query_times.push(query_time);

            progress_bar.inc(1);
        }
        let total_search_time = search_start.elapsed();
        info!("Search ended");

        let conn = Connection::open(DB_PATH)?;
        save_puffinn_results(
            &conn,
            config,
            n,
            memory,
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
        memory_used_bytes: usize,
        total_search_time: Duration,
        puffinn_counts: &[u32],
        query_times: &[Duration],
    ) -> Result<(), rusqlite::Error> {
        // Insert overall results
        conn.execute(
            "INSERT OR REPLACE INTO puffinn_results 
        (num_tables, k, delta, dataset, dataset_len, memory_used_bytes, 
         total_time_ms, queries_per_second) 
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                config.num_tables,
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
        (num_tables, k, delta, dataset, query_idx, query_time_ms, distance_computations) 
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;

        for (idx, (count, query_time)) in puffinn_counts.iter().zip(query_times.iter()).enumerate()
        {
            stmt.execute(params![
                config.num_tables,
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
            let dataset_path = format!("./datasets/{}.hdf5", config.dataset_name);
            let hdf5_dataset = load_hdf5_dataset(&dataset_path)?;

            let data = AngularData::new(hdf5_dataset.dataset_array);

            // run clann
            match check_configuration_exists_clann(&conn, config, git_hash) {
                Ok(false) => {
                    // Run benchmark, catching any errors for this specific configuration
                    match run_benchmark_config_clann(
                        config,
                        data.clone(),
                        &hdf5_dataset.dataset_queries,
                        &hdf5_dataset.ground_truth_distances,
                        config_idx,
                    ) {
                        Ok(_) => {
                            info!("CLANN config {} run", config_idx);
                        }
                        Err(e) => {
                            error!(
                                "Error running benchmark for configuration {}: {}",
                                config_idx, e
                            );
                        }
                    }
                }
                Ok(true) => {
                    warn!("Configuration {} already exists (unexpected)", config_idx);
                }
                Err(BenchmarkError::ConfigExists(_msg)) => {
                    info!("Skipping configuration {} for CLANN", config_idx);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }

            match check_configuration_exists_puffinn(&conn, config) {
                Ok(false) => {
                    // run puffinn
                    match run_benchmark_config_puffinn(
                        config,
                        &data,
                        &hdf5_dataset.dataset_queries,
                        config_idx,
                    ) {
                        Ok(_) => {
                            info!("PUFFINN config {} run", config_idx);
                        }
                        Err(e) => {
                            error!(
                                "Error running puffinn benchmark for configuration {}: {}",
                                config_idx, e
                            );
                        }
                    }
                }
                Ok(true) => {
                    info!("Skipping configuration {} for PUFFINN", config_idx);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }

        Ok(())
    }

    pub fn run_distance_benchmarks(_c: &mut Criterion) {
        env_logger::Builder::from_env(Env::default().default_filter_or("info"))
            .format_timestamp_millis()
            .init();

        print_benchmark_header("PUFFINN-CLANN Distance Computations Comparison");
        compare_implementations_distance().expect("Error in compare implem");
    }

    criterion_group! {
        name = distance_benches;
        config = Criterion::default().configure_from_args();
        targets = run_distance_benchmarks
    }

    criterion_main!(distance_benches);
