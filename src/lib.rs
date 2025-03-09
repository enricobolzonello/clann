//! Clustered LSH-based Algorithm for the Nearest Neighbors problem
//!
//! CLANN is an algorithm solving the Nearest Neighbors problem, built on top of PUFFINN. Rather than constructing a single index, the algorithm begins by dividing the dataset into clusters based on a user-defined number. It then builds a separate PUFFINN index for each cluster.
//! This approach, even though requires more memory and index building time, effectively cuts the hit distribution for the LSH function, ensuring that points that are far apart cannot collide. In classic LSH scenarios, it has been observed long tails of hits, due to the probabilistic nature of the function. Even though far points have low probability of colliding it was still not null, and the problem accentuated with queries far away from the dataset, where it approximates to a brute-force approach.
//!

use core::{config::MetricsGranularity, index::ClusteredIndex, Config, Result};
use std::time::Duration;

use metricdata::{MetricData, Subset};
use ndarray::{Array, Ix2};
use puffinn_binds::IndexableSimilarity;

pub mod core;
pub mod metricdata;
pub mod puffinn_binds;
pub mod utils;

/// Initializes a CLANN index from a previously serialized file.
///
/// # Parameters
/// - `data`: Dataset to search over, must match the original dataset used to build the index
/// - `file_path`: Path to the HDF5 file containing the serialized index
///
/// # Returns
/// A `ClusteredIndex` instance loaded from the file, ready to be used for searching
///
/// # Errors
/// Returns `ClusteredIndexError::ConfigError` if:
/// - The file doesn't exist
/// - The file format is invalid
/// - The serialized data is corrupted or incompatible
///
/// # Example
/// ```no_run
/// use clann::{init_from_file, metricdata::AngularData};
/// 
/// let data = AngularData::new(/* your dataset */);
/// let index = init_from_file(data, "path/to/index.h5").unwrap();
/// ```
pub fn init_from_file<T>(data: T, file_path: &str) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    ClusteredIndex::new_from_file(data, file_path)
}

/// Initializes a new CLANN index with default configuration.
///
/// Default configuration uses:
/// - 10 table for PUFFINN indices
/// - Clustering factor of 1.0 (sqrt(n) clusters)
/// - k = 10 nearest neighbors
/// - delta = 0.9 (target recall)
/// - No metrics collection
///
/// # Parameters
/// - `data`: Dataset to build the index for
///
/// # Returns
/// An unbuilt `ClusteredIndex` instance with default configuration.
/// Call [`build()`] to construct the index before searching.
///
/// # Errors
/// Returns `ClusteredIndexError::DataError` if the input dataset is empty
///
/// # Example
/// ```no_run
/// use clann::{init, build, metricdata::AngularData};
/// 
/// let data = AngularData::new(/* your dataset */);
/// let mut index = init(data).unwrap();
/// build(&mut index).unwrap();
/// ```
pub fn init<T>(data: T) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    init_with_config(data, Config::default())
}

/// Initializes a new CLANN index with custom configuration.
///
/// # Parameters
/// - `data`: Dataset to build the index for
/// - `config`: Configuration object specifying:
///   - Number of tables for PUFFINN indices
///   - Clustering factor (determines number of clusters as sqrt(n) * factor)
///   - k nearest neighbors to search for
///   - Target recall (delta)
///   - Dataset name and metrics configuration
///
/// # Returns
/// An unbuilt `ClusteredIndex` instance with the specified configuration.
/// Call [`build()`] to construct the index before searching.
///
/// # Errors
/// Returns `ClusteredIndexError::DataError` if the input dataset is empty
///
/// # Example
/// ```no_run
/// use clann::{init_with_config, build, core::Config, metricdata::AngularData};
/// 
/// let data = AngularData::new(/* your dataset */);
/// let config = Config::new(
///     84,     // num_tables
///     0.4,    // clustering_factor
///     10,     // k
///     0.9,    // delta
///     "glove", // dataset_name
///     MetricsOutput::DB // metrics output
/// );
/// let mut index = init_with_config(data, config).unwrap();
/// build(&mut index).unwrap();
/// ```
pub fn init_with_config<T>(data: T, config: Config) -> Result<ClusteredIndex<T>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    ClusteredIndex::new(config, data)
}

/// Builds a CLANN index by performing clustering and creating PUFFINN indices.
///
/// The build process consists of two main steps:
/// 1. Clustering: Uses greedy minimum-maximum clustering to partition the dataset
/// 2. Index Creation: Creates a PUFFINN index for each cluster (except small ones which use brute force)
///
/// # Parameters
/// - `index`: Unbuilt index instance to build
///
/// # Performance
/// - Time complexity: O(n * sqrt(n)) for clustering + O(n * L) for PUFFINN index creation
/// - Space complexity: O(n) for cluster assignments + O(n * L) for PUFFINN indices
/// where n is the dataset size and L is the number of tables
///
/// # Errors
/// Returns `ClusteredIndexError::PuffinnCreationError` if PUFFINN index creation fails for any cluster
pub fn build<T>(index: &mut ClusteredIndex<T>) -> Result<()>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.build()
}

/// Searches for the k nearest neighbors of a query point.
///
/// The search process:
/// 1. Sorts clusters by distance from query to their centers
/// 2. Processes clusters in order until termination condition is met
/// 3. For each cluster either:
///    - Uses PUFFINN index to find candidates (large clusters)
///    - Uses brute force search (small clusters)
///
/// # Parameters
/// - `index`: Built index to search in
/// - `query`: Query point with same dimensionality as dataset points
///
/// # Returns
/// Vector of (distance, index) pairs for the k nearest neighbors found,
/// sorted by distance in ascending order
///
/// # Errors
/// - `ClusteredIndexError::IndexNotFound` if a required PUFFINN index is missing
/// - `ClusteredIndexError::PuffinnSearchError` if PUFFINN search fails
/// - `ClusteredIndexError::IndexOutOfBounds` if candidate mapping fails
///
/// # Example
/// ```no_run
/// use clann::{init, build, search, metricdata::AngularData};
/// 
/// let data = AngularData::new(/* your dataset */);
/// let mut index = init(data).unwrap();
/// build(&mut index).unwrap();
/// 
/// let query = vec![0.1, 0.2, 0.3];
/// let neighbors = search(&mut index, &query).unwrap();
/// ```
pub fn search<T>(index: &mut ClusteredIndex<T>, query: &[T::DataType]) -> Result<Vec<(f32, usize)>>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    index.search(query)
}

/// Saves metrics from a search run to a SQLite database.
///
/// # Parameters
/// - `index`: Index containing the metrics to save
/// - `output_path`: Path to SQLite database file
/// - `granularity`: Level of detail for metrics:
///   - `Run`: Only overall metrics like recall and total time
///   - `Query`: Run metrics + per-query metrics
///   - `Cluster`: Query metrics + per-cluster metrics
/// - `ground_truth_distances`: True k-NN distances for computing recall
/// - `run_distances`: Distances returned by the search algorithm
/// - `total_search_time`: Total time spent on all queries
///
/// # Database Schema
/// The metrics are saved in multiple tables:
/// - `build_metrics`: Index building statistics
/// - `search_metrics`: Overall search performance
/// - `search_metrics_query`: Per-query metrics
/// - `search_metrics_cluster`: Per-cluster metrics
///
/// # Errors
/// - `ClusteredIndexError::MetricsError` if metrics are not enabled or database doesn't exist
/// - `ClusteredIndexError::ResultDBError` for database connection/operation errors
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

/// Serializes a CLANN index to an HDF5 file.
///
/// # Parameters
/// - `index`: Index to serialize
/// - `directory_path`: Directory where the index file will be saved
///
/// # File Structure
/// The HDF5 file contains:
/// - Configuration parameters
/// - Cluster information (centers, assignments, radii)
/// - PUFFINN indices for each cluster
///
/// # File Naming
/// The file is named: `index_{dataset_name}_k{clusters_factor}_L{num_tables}.h5`
///
/// # Errors
/// Returns `ClusteredIndexError::SerializeError` if:
/// - Directory doesn't exist
/// - File creation fails
/// - Serialization of any component fails
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
