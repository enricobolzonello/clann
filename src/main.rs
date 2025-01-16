use std::time::Instant;

use clann::{build, core::Config, enable_metrics, init_with_config, metricdata::AngularData, save_metrics, search, utils::load_hdf5_dataset};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();

    info!("Starting search benchmark");
    let total_start = Instant::now();

    const OUTPUT_PATH: &str = "/home/bolzo/puffinn-tests/output";

    let (data_raw, queries) = load_hdf5_dataset("/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5").unwrap();
    let data = AngularData::new(data_raw);

    let config = Config{
        memory_limit: 2*1073741824,
        num_clusters: 4,
        k: 10,
        delta: 0.9,
    };

    let mut index = init_with_config(data, config).unwrap();

    build(&mut index).map_err(|e| eprintln!("Error: {}", e)).unwrap();

    enable_metrics(&mut index).unwrap();

    info!("Processing {} queries", queries.nrows());
    let search_start = Instant::now();
    let mut distance_results = Vec::with_capacity(queries.nrows());
    
    let progress_bar = ProgressBar::new(queries.nrows() as u64);
    progress_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Failed to set progress bar style")
        .progress_chars("=>-"));

    let mut total_search_time = std::time::Duration::new(0, 0);
    let mut min_search_time = std::time::Duration::new(u64::MAX, 0);
    let mut max_search_time = std::time::Duration::new(0, 0);

    for (i, query) in queries.rows().into_iter().enumerate() {
        let query_start = Instant::now();
        
        let result = search(&mut index, query.as_slice().unwrap()).unwrap();
        
        let query_time = query_start.elapsed();
        total_search_time += query_time;
        min_search_time = min_search_time.min(query_time);
        max_search_time = max_search_time.max(query_time);

        let distances: Vec<f32> = result.iter()
            .map(|&(distance, _)| distance)
            .collect();
        distance_results.push(distances);

        if (i + 1) % 1000 == 0 {
            let avg_time = total_search_time / (i as u32 + 1);
            progress_bar.set_message(format!(
                "Avg: {:?} Min: {:?} Max: {:?}", 
                avg_time, min_search_time, max_search_time
            ));
        }
        
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Search complete");
    
    let total_search_time = search_start.elapsed();
    info!("All queries processed in {:?}", total_search_time);
    info!("Average query time: {:?}", total_search_time / queries.nrows() as u32);
    info!("Min query time: {:?}", min_search_time);
    info!("Max query time: {:?}", max_search_time);
    info!("Total results: {}", distance_results.len());

    // Save metrics
    info!("Saving metrics to {}", OUTPUT_PATH);
    save_metrics(&mut index, OUTPUT_PATH).unwrap();

    info!("Benchmark completed in {:?}", total_start.elapsed());
}
