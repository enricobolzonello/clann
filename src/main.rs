use std::{env, fs, time::{Duration, Instant}};

use clann::{build, core::Config, init_from_file, init_with_config, metricdata::AngularData, save_metrics, search, serialize, utils::{load_hdf5_dataset, MetricsGranularity, MetricsOutput}};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = env::args().collect();
    info!("Starting search benchmark");
    let total_start = Instant::now();

    const DB_PATH: &str = "./clann_results.sqlite3";
    const INDEX_DIR: &str = "./__index_cache__";

    let hdf5_dataset = load_hdf5_dataset("./datasets/glove-25-angular.hdf5").unwrap();
    let data = AngularData::new(hdf5_dataset.dataset_array);

    let config = Config{
        num_tables: 50,
        num_clusters_factor: 0.4,
        k: 10,
        delta: 0.9,
        dataset_name: "glove-25-angular".to_owned(),
        metrics_output: MetricsOutput::DB,
    };

    let index_path = format!(
        "{}/index_{}_k{:.2}_L{}.h5",
        INDEX_DIR, config.dataset_name, config.num_clusters_factor, config.num_tables
    );

    let mut index = if fs::metadata(&index_path).is_ok() {
        info!("Loading index from file: {}", index_path);
        init_from_file(data, &index_path).unwrap()
    } else {
        info!("No saved index found, initializing a new one");
        let mut new_index = init_with_config(data, config).unwrap();
        build(&mut new_index).map_err(|e| eprintln!("Error: {}", e)).unwrap();
        serialize(&new_index, INDEX_DIR).unwrap();
        new_index
    };

    info!("Processing {} queries", hdf5_dataset.dataset_queries.nrows());
    let mut distance_results = Vec::with_capacity(hdf5_dataset.dataset_queries.nrows());
    
    let progress_bar = ProgressBar::new(hdf5_dataset.dataset_queries.nrows() as u64);
    progress_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Failed to set progress bar style")
        .progress_chars("=>-"));

    let mut total_queries = 0u32;
    let mut min_search_time = Duration::from_secs(u64::MAX);
    let mut max_search_time = Duration::from_secs(0);
    let mut total_search_time = Duration::from_secs(0);

    for (i, query) in hdf5_dataset.dataset_queries.rows().into_iter().enumerate() {
        let query_start = Instant::now();
        let result = search(&mut index, query.as_slice().unwrap()).unwrap();
        let query_time = query_start.elapsed();

        total_queries += 1;
        total_search_time += query_time;
        min_search_time = min_search_time.min(query_time);
        max_search_time = max_search_time.max(query_time);

        let distances: Vec<f32> = result.iter()
            .map(|&(distance, _)| distance)
            .collect();
        distance_results.push(distances);

        if (i + 1) % 1000 == 0 {
            progress_bar.set_message(format!(
                "Min: {:?} Max: {:?}", 
                min_search_time, max_search_time
            ));
        }
        
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Search complete");

    let average_time = total_search_time / total_queries;

    info!("All queries processed in {:?}", total_search_time);
    info!("Average query time: {:?}", average_time);
    info!("Min query time: {:?}", min_search_time);
    info!("Max query time: {:?}", max_search_time);
    info!("Total results: {}", distance_results.len());

    if args.len() > 1 && &args[1] == "--save" {
        info!("Saving metrics to {}", DB_PATH);
        save_metrics(&mut index, 
            DB_PATH,
            MetricsGranularity::Cluster,
            &hdf5_dataset.ground_truth_distances,
            &distance_results,
            &total_search_time
        ).unwrap();
    }

    info!("Benchmark completed in {:?}", total_start.elapsed());
}