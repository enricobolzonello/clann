//! Clustered LSH-based Algorithm for the Nearest Neighbors problem
//!
//! CLANN is an algorithm solving the Nearest Neighbors problem, built on top of PUFFINN. Rather than constructing a single index, the algorithm begins by dividing the dataset into clusters based on a user-defined number. It then builds a separate PUFFINN index for each cluster.
//! This approach, even though requires more memory and index building time, effectively cuts the hit distribution for the LSH function, ensuring that points that are far apart cannot collide. In classic LSH scenarios, it has been observed long tails of hits, due to the probabilistic nature of the function. Even though far points have low probability of colliding it was still not null, and the problem accentuated with queries far away from the dataset, where it approximates to a brute-force approach.
//!

use core::{index::ClusteredIndex, Config, Result};
use std::time::Duration;

use metricdata::{MetricData, Subset};
use ndarray::{Array, Ix2};
use puffinn_binds::IndexableSimilarity;
use utils::MetricsGranularity;

pub mod core;
pub mod metricdata;
pub mod puffinn_binds;
pub mod utils;

pub fn init_from_file<T>(data: T, file_path: &str) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    ClusteredIndex::new_from_file(data, file_path)
}

pub fn init<T>(data: T) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    init_with_config(data, Config::default())
}

pub fn init_with_config<T>(data: T, config: Config) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    ClusteredIndex::new(config, data)
}

pub fn build<T>(index: &mut ClusteredIndex<T>) -> Result<()>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.build()
}

pub fn search<T>(index: &mut ClusteredIndex<T>, query: &[T::DataType]) -> Result<Vec<(f32, usize)>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.search(query)
}

pub fn enable_run_metrics<T>(index: &mut ClusteredIndex<T>) -> Result<()>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.enable_metrics()
}

pub fn save_metrics<T>(
    index: &mut ClusteredIndex<T>,
    output_path: &str,
    granularity: MetricsGranularity,
    ground_truth_distances: &Array<f32, Ix2>,
    run_distances: &[Vec<f32>],
    total_search_time: &Duration,
) -> Result<()>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.save_metrics(
        output_path.to_string(),
        granularity,
        ground_truth_distances,
        run_distances,
        total_search_time,
    )
}

pub fn serialize<T>(
    index: &ClusteredIndex<T>,
    directory_path: &str,
) -> Result<()>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.serialize(directory_path)
}
