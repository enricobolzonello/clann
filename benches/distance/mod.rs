use clann::puffinn_binds::puffinn_index::get_distance_computations;
use clann::{
    build, init_with_config, metricdata::AngularData, puffinn_binds::PuffinnIndex,
    search, utils::load_hdf5_dataset,
};
use criterion::Criterion;
use rand::{seq::SliceRandom, thread_rng};

use crate::utils::CONFIGS;

pub fn compare_implementations_distance(_c: &Criterion, dataset_path: &str) {
    // Load dataset
    let (data_raw, queries) = load_hdf5_dataset(dataset_path).unwrap();

    // Select queries
    let num_queries = queries.nrows();
    let mut rng = thread_rng();
    let query_indices: Vec<usize> = (0..num_queries)
        .collect::<Vec<usize>>()
        .choose_multiple(&mut rng, 10)
        .cloned()
        .collect();

    for (config_idx, config) in CONFIGS.iter().enumerate() {
        let data = AngularData::new(data_raw.clone());

        // Initialize base PUFFINN index
        let base_index = PuffinnIndex::new(&data, config.memory_limit).unwrap();

        // Initialize clustered index with metrics enabled
        let mut clustered_index = init_with_config(data, config.clone()).unwrap();
        let _ = clustered_index.enable_metrics();
        build(&mut clustered_index).unwrap();

        println!("\nDistance Computation Comparison for config {}:", config_idx);
        println!("{:<10} {:<15} {:<15}", "Query ID", "Clustered", "PUFFINN");
        println!("{:-<40}", "");

        for &query_idx in &query_indices {
            let query = queries.row(query_idx);
            let query_slice = query.as_slice().unwrap();

            // Clustered implementation
            search(&mut clustered_index, query_slice).unwrap();
            let clustered_count = clustered_index
                .metrics
                .as_ref()
                .unwrap()
                .current_query()
                .distance_computations;

            // Base PUFFINN implementation
            base_index
                .search::<AngularData<ndarray::OwnedRepr<f32>>>(query_slice, config.k, config.delta)
                .unwrap();
            let puffinn_count = get_distance_computations();

            println!(
                "{:<10} {:<15} {:<15}",
                query_idx, clustered_count, puffinn_count
            );
        }
    }
}
