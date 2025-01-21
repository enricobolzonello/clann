use clann::metricdata::MetricData;
use clann::puffinn_binds::puffinn_index::get_distance_computations;
use clann::{
    build, init_with_config, metricdata::AngularData, puffinn_binds::PuffinnIndex,
    search, utils::load_hdf5_dataset,
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::fs::File;
use std::io::Write;
use utils::{create_progress_bar, load_configs_from_file, print_benchmark_header, DATASET_PATH};

mod utils;

pub fn export_to_csv(
    clustered_data: &[Vec<u32>],
    puffinn_data: &[Vec<u32>],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    
    writeln!(file, "config_id,query_id,method,value")?;
    
    for (config_id, data) in clustered_data.iter().enumerate() {
        for (query_id, &value) in data.iter().enumerate() {
            writeln!(file, "{},{},clustered,{}", config_id, query_id, value)?;
        }
    }
    
    for (config_id, data) in puffinn_data.iter().enumerate() {
        for (query_id, &value) in data.iter().enumerate() {
            writeln!(file, "{},{},puffinn,{}", config_id, query_id, value)?;
        }
    }
    
    Ok(())
}

pub fn compare_implementations_distance(dataset_path: &str) {
    let configs = load_configs_from_file("benches/configs.json").unwrap();

    let (data_raw, queries, _) = load_hdf5_dataset(dataset_path).expect("Failed to load dataset");

    println!("\nDistance Computation Comparison Summary:");
    println!("{:-<120}", "");
    println!("{:<10} {:<15} {:<15} {:<15} {:<15} {:<20} {:<20}", 
        "Config", "Clustered Avg", "PUFFINN Avg", "Clustered Std", "Improvement %", 
        "Min (C/P)", "Max (C/P)");
    println!("{:-<120}", "");

    let mut clustered_counts_all = Vec::new();
    let mut puffinn_counts_all = Vec::new();

    for (config_idx, config) in configs.iter().enumerate() {
        let data = AngularData::new(data_raw.clone());

        let base_index = PuffinnIndex::new(&data, config.kb_per_point * data.num_points() * 1024).expect("Failed to initialize PUFFINN index");
        let mut clustered_index = init_with_config(data, config.clone()).expect("Failed to initialize clustered index");
        clustered_index.enable_metrics().expect("Failed to enable metrics");
        build(&mut clustered_index).expect("Failed to build clustered index");

        let mut clustered_counts = Vec::new();
        let mut puffinn_counts = Vec::new();

        for query in queries.rows() {
            let query_slice = query.as_slice().expect("Failed to get query slice");

            search(&mut clustered_index, query_slice).expect("Clustered search failed");
            let clustered_count = clustered_index
                .metrics
                .as_ref()
                .expect("Metrics not enabled")
                .current_query()
                .distance_computations as u32;

            base_index
                .search::<AngularData<ndarray::OwnedRepr<f32>>>(query_slice, config.k, config.delta)
                .expect("PUFFINN search failed");
            let puffinn_count = get_distance_computations();

            clustered_counts.push(clustered_count);
            puffinn_counts.push(puffinn_count);
        }

        clustered_counts_all.push(clustered_counts.clone());
        puffinn_counts_all.push(puffinn_counts.clone());

        let clustered_avg = clustered_counts.iter().sum::<u32>() as f64 / clustered_counts.len() as f64;
        let puffinn_avg = puffinn_counts.iter().sum::<u32>() as f64 / puffinn_counts.len() as f64;

        let clustered_std = (clustered_counts.iter()
            .map(|&x| (x as f64 - clustered_avg).powi(2))
            .sum::<f64>() / clustered_counts.len() as f64)
            .sqrt();

        let improvement = ((puffinn_avg - clustered_avg) / puffinn_avg * 100.0).max(-999.9);

        let min_c = clustered_counts.iter().min().unwrap_or(&0);
        let min_p = puffinn_counts.iter().min().unwrap_or(&0);
        let max_c = clustered_counts.iter().max().unwrap_or(&0);
        let max_p = puffinn_counts.iter().max().unwrap_or(&0);

        println!("{:<10} {:<15.2} {:<15.2} {:<15.2} {:<15.2}% {:<20} {:<20}", 
            config_idx, clustered_avg, puffinn_avg, clustered_std, improvement, 
            format!("{}/{}", min_c, min_p),
            format!("{}/{}", max_c, max_p));
    }

    export_to_csv(
        &clustered_counts_all, 
        &puffinn_counts_all, 
        "./runs/distance_computations.csv"
    ).expect("Failed to export data to CSV");
    
    println!("\nData has been exported to distance_computations.csv");
}

pub fn run_distance_benchmarks(_c: &mut Criterion) {
    print_benchmark_header("PUFFINN-CLANN Distance Computations Comparison");
    let pb = create_progress_bar("Running distance comparison".to_string(), 100);
    compare_implementations_distance(DATASET_PATH);
    pb.finish_with_message("Distance Comparison complete");
}

criterion_group! {
    name = distance_benches;
    config = Criterion::default().configure_from_args();
    targets = run_distance_benchmarks
}

criterion_main!(distance_benches);