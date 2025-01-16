use clann::{build, core::Config, init_with_config, metricdata::AngularData, puffinn_binds::PuffinnIndex, search, utils::load_hdf5_dataset};
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, AxisScale
};
use std::time::Duration;
use rand::{seq::SliceRandom, thread_rng};


const DATASET_PATH: &str = "/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5";
const MEMORY_LIMIT: usize = 1 * 1024 * 1024 * 1024; // 1GB
const K: usize = 10;
const DELTA: f32 = 0.9;

fn compare_implementations(c: &mut Criterion) {
    // Load dataset
    let (data_raw, queries) = load_hdf5_dataset(DATASET_PATH).unwrap();
    let data = AngularData::new(data_raw.clone());
    
    // Initialize clustered index
    let config = Config {
        memory_limit: MEMORY_LIMIT,
        num_clusters: 4,
        k: K,
        delta: DELTA,
    };
    let mut clustered_index = init_with_config(data, config).unwrap();
    build(&mut clustered_index).unwrap();
    
    // Initialize base PUFFINN index
    let base_index = PuffinnIndex::new(&AngularData::new(data_raw), MEMORY_LIMIT)
        .unwrap();
    
    // Configure benchmark group
    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);
    
    let mut group = c.benchmark_group("knn_search");
    group
        .plot_config(plot_config)
        .sample_size(50)
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(5));

    // Select a subset of queries for benchmarking
    let num_queries = queries.nrows();
    let mut rng = thread_rng();
    let query_indices: Vec<usize> = (0..num_queries)
        .collect::<Vec<usize>>()
        .choose_multiple(&mut rng, 10)
        .cloned()
        .collect();
    
    for &query_idx in &query_indices {
        let query = queries.row(query_idx);
        let query_slice = query.as_slice().unwrap();
        
        // Benchmark clustered implementation
        group.bench_with_input(
            BenchmarkId::new("clustered", query_idx), 
            &query_slice,
            |b, q| {
                b.iter(|| {
                    search(&mut clustered_index, q).unwrap()
                });
            },
        );
        
        // Benchmark base implementation
        group.bench_with_input(
            BenchmarkId::new("base_puffinn", query_idx),
            query_slice,
            |b, q| {
                b.iter(|| {
                    base_index.search::<AngularData<ndarray::OwnedRepr<f32>>>(q, K, DELTA).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_plots()
        .sample_size(50)
        .measurement_time(Duration::from_secs(30));
    targets = compare_implementations
}
criterion_main!(benches);