use clann::{
    build,
    core::Config,
    init_with_config,
    metricdata::{AngularData, MetricData},
    puffinn_binds::PuffinnIndex,
    search,
    utils::load_hdf5_dataset,
};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration
};
use rand::{seq::SliceRandom, thread_rng};
use utils::{create_progress_bar, load_configs_from_file, print_benchmark_header};
use core::f32;
use std::time::Duration;

mod utils;

pub fn compare_implementations_time(c: &mut Criterion) {
    let configs = load_configs_from_file("benches/configs.json").unwrap();

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);

    let dataset_path = format!("./datasets/{}.hdf5", configs[0].dataset_name);      // assume the dataset does not change
    let hdf5_dataset = load_hdf5_dataset(&dataset_path).unwrap();

    // Select a subset of queries for benchmarking
    let num_queries = hdf5_dataset.dataset_queries.nrows();
    let mut rng = thread_rng();
    let query_indices: Vec<usize> = (0..num_queries)
        .collect::<Vec<usize>>()
        .choose_multiple(&mut rng, 10)
        .cloned()
        .collect();

    for (config_idx, config) in configs.iter().enumerate() {
        let data = AngularData::new(hdf5_dataset.dataset_array.clone());

        // Initialize base PUFFINN index
        let (base_index, _memory) = PuffinnIndex::new(&data, config.num_tables * data.num_points() * 1024).unwrap();

        // Initialize clustered index
        let clann_config = Config {
            num_tables: config.num_tables,
            num_clusters_factor: config.num_clusters_factor,
            k: config.k,
            delta: config.delta,
            dataset_name: config.dataset_name.clone(),
            metrics_output: clann::utils::MetricsOutput::DB,
        };
        let mut clustered_index = init_with_config(data, clann_config).unwrap();
        build(&mut clustered_index).unwrap();

        let group_name = format!(
            "config_{}_clusters_{}_L_{}_dataset_{}",
            config_idx,
            config.num_clusters_factor,
            config.num_tables,
            dataset_path.split('/').last().unwrap_or("unknown")
        );

        let mut group = c.benchmark_group(&group_name);
        group
            .plot_config(plot_config.clone())
            .sample_size(15)
            .measurement_time(Duration::from_secs(10))
            .warm_up_time(Duration::from_secs(1));

        for &query_idx in &query_indices {
            let query = hdf5_dataset.dataset_queries.row(query_idx);
            let query_slice = query.as_slice().unwrap();

            // Benchmark clustered implementation
            group.bench_with_input(
                BenchmarkId::new("clustered", query_idx),
                &query_slice,
                |b, q| {
                    b.iter(|| search(&mut clustered_index, q).unwrap());
                },
            );

            // Benchmark base implementation
            group.bench_with_input(
                BenchmarkId::new("base_puffinn", query_idx),
                query_slice,
                |b, q| {
                    b.iter(|| {
                        base_index
                            .search::<AngularData<ndarray::OwnedRepr<f32>>>(q, config.k, f32::INFINITY, config.delta)
                            .unwrap()
                    });
                },
            );
        }

        group.finish();
    }
}

pub fn run_time_benchmarks(c: &mut Criterion) {
    print_benchmark_header("PUFFINN-CLANN Time Comparison");
    let pb = create_progress_bar("Running time comparison".to_string(), 100);
    compare_implementations_time(c);
    pb.finish_with_message("Time Comparison complete");
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

criterion_main!(time_benches);