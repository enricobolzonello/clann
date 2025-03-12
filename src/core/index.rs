use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use hdf5::types::{VarLenAscii, VarLenUnicode};
use hdf5::File;
use log::{debug, error, info, trace};
use ndarray::{Array, Ix2};
use ordered_float::OrderedFloat;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::core::config::MetricsOutput;
use crate::core::heap::Element;
use crate::core::{ClusteredIndexError, Config, Result};
use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::get_distance_computations;
use crate::puffinn_binds::puffinn::clear_distance_computations;
use crate::puffinn_binds::IndexableSimilarity;
use crate::puffinn_binds::PuffinnIndex;
use crate::utils::{db_exists, RunMetrics};

use super::config::MetricsGranularity;
use super::gmm::greedy_minimum_maximum;
use super::heap::TopKClosestHeap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ClusterCenter {
    pub(crate) idx: usize, // index of the cluster, corresponds to the index of the vec of puffinn indexes
    pub(crate) center_idx: usize, // index of the center point in the original dataset
    pub(crate) radius: f32, // radius of the cluster
    pub(crate) assignment: Vec<usize>, // vector of indices to the original dataset for points assigned to this cluster
    pub(crate) brute_force: bool, // flag indicating if brute force is applied instead of puffinn (<500 points)
    pub(crate) memory_used: usize, // memory used by the puffinn index
}

pub struct ClusteredIndex<T>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    data: T,
    clusters: Vec<ClusterCenter>,
    config: Config,
    puffinn_indices: Vec<Option<PuffinnIndex>>,
    pub(crate) metrics: Option<RunMetrics>,
}

impl<T> ClusteredIndex<T>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    /// Creates a new Clustered Index from scratch.
    ///
    /// # Parameters
    /// - `config`: Configuration object specifying:
    ///   - Number of tables for PUFFINN indices
    ///   - Clustering factor (determines number of clusters as sqrt(n) * factor)
    ///   - k nearest neighbors to search for
    ///   - Target recall (delta)
    ///   - Dataset name and metrics configuration
    /// - `data`: The dataset implementing required traits for distance computation and subset operations
    ///
    /// # Returns
    /// A new `ClusteredIndex` instance initialized with the given configuration but not yet built.
    /// The index needs to be built using [`build()`] before it can be used for searching.
    ///
    /// # Errors
    /// Returns `ClusteredIndexError::DataError` if the input dataset is empty
    pub(crate) fn new(config: Config, data: T) -> Result<Self> {
        if data.num_points() == 0 {
            return Err(ClusteredIndexError::DataError("empty dataset".to_string()));
        }

        info!("Initializing Index with config {:?}", config);

        let k = ((config.num_clusters_factor as f64 * (data.num_points() as f64).sqrt()).floor()
            as usize)
            .max(1);
        let metrics = matches!(config.metrics_output, MetricsOutput::DB)
            .then(|| RunMetrics::new(config.clone(), data.num_points()));

        Ok(ClusteredIndex {
            data,
            clusters: Vec::with_capacity(k),
            config,
            puffinn_indices: Vec::with_capacity(k),
            metrics,
        })
    }

    /// Creates a new Clustered Index by loading a previously serialized index from a file.
    ///
    /// # Parameters
    /// - `data`: The dataset implementing required traits, must match the original dataset used to build the index
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
    pub(crate) fn new_from_file(data: T, file_path: &str) -> Result<Self> {
        if !Path::new(file_path).exists() {
            return Err(ClusteredIndexError::ConfigError(format!(
                "file {} not found",
                file_path
            )));
        }

        let file =
            File::open(file_path).map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let root = file
            .group("/")
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;

        // read config
        let config_dataset = root
            .dataset("config")
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let config_ascii = config_dataset
            .read_scalar::<VarLenAscii>()
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let config: Config = serde_json::from_str(config_ascii.as_str())
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let metrics = matches!(config.metrics_output, MetricsOutput::DB)
            .then(|| RunMetrics::new(config.clone(), data.num_points()));

        // read cluster centers
        let cluster_dataset = root
            .dataset("clusters")
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let cluster_ascii = cluster_dataset
            .read_scalar::<VarLenAscii>()
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;
        let clusters: Vec<ClusterCenter> = serde_json::from_str(cluster_ascii.as_str())
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()))?;

        // read puffinn indices
        let mut puffinn_indices = Vec::new();
        for c in &clusters {
            if !c.brute_force {
                let index =
                    PuffinnIndex::new_from_file(file_path, &format!("index_{}", c.idx)).unwrap();
                puffinn_indices.push(Some(index));
            } else {
                puffinn_indices.push(None);
            }
        }

        Ok(Self {
            data,
            clusters,
            config,
            puffinn_indices,
            metrics,
        })
    }

    /// Builds the index by performing clustering and creating PUFFINN indices.
    ///
    /// The build process consists of two main steps:
    /// 1. Clustering: Uses greedy minimum-maximum clustering to partition the dataset
    /// 2. Index Creation: Creates a PUFFINN index for each cluster (except small ones which use brute force)
    ///
    /// # Performance
    /// - Time complexity: O(n * sqrt(n)) for clustering + O(n * L) for PUFFINN index creation
    /// - Space complexity: O(n) for cluster assignments + O(n * L) for PUFFINN indices
    /// where n is the dataset size and L is the number of tables
    ///
    /// # Errors
    /// Returns `ClusteredIndexError::PuffinnCreationError` if PUFFINN index creation fails for any cluster
    pub(crate) fn build(&mut self) -> Result<()> {
        let total_clusters = self.clusters.capacity();
        info!("Starting build process with {} clusters", total_clusters);

        // 1) PERFORM CLUSTERING
        info!("Performing greedy clustering...");
        let start_clustering = std::time::Instant::now();
        let (centers, assignment, radius) =
            greedy_minimum_maximum(&self.data, self.clusters.capacity());
        info!("Clustering completed in {:.2?}", start_clustering.elapsed());

        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];

        for (data_idx, &center_pos) in assignment.iter().enumerate() {
            assignments[center_pos].push(data_idx);
        }

        self.clusters = centers
            .iter()
            .zip(radius.iter())
            .zip(assignments)
            .enumerate()
            .map(|(idx, ((&center_idx, &radius), assignment_indexes))| {
                let cluster = ClusterCenter {
                    idx,
                    center_idx,
                    radius,
                    brute_force: assignment_indexes.len() < 100
                        || assignment_indexes.len() < self.config.k,
                    assignment: assignment_indexes,
                    memory_used: 0,
                };

                trace!(
                    "Cluster {}: center_idx={}, points={}, radius={}",
                    idx,
                    cluster.center_idx,
                    cluster.assignment.len(),
                    cluster.radius,
                );

                cluster
            })
            .collect();

        // 2) CREATE PUFFINN INDEXES
        info!("Creating Puffinn indexes...");
        self.puffinn_indices = Vec::with_capacity(self.clusters.len());
        for (cluster_idx, cluster) in self.clusters.iter_mut().enumerate() {
            // Progress logging
            if cluster_idx % 10 == 0 {
                info!(
                    "Processing cluster {}/{} ({}%)",
                    cluster_idx + 1,
                    total_clusters,
                    ((cluster_idx + 1) as f32 / total_clusters as f32 * 100.0).round()
                );
            }

            if cluster.assignment.is_empty() {
                debug!("Skipping empty cluster {}", cluster_idx);
                continue;
            }

            if cluster.brute_force {
                info!(
                    "Skipping cluster {} with {} points: doing brute force",
                    cluster.idx,
                    cluster.assignment.len()
                );
                self.puffinn_indices.push(None);
                continue;
            }

            debug!(
                "Cluster {}: L {}, points: {}",
                cluster_idx,
                self.config.num_tables,
                cluster.assignment.len()
            );

            // Create Puffinn index
            match PuffinnIndex::new(
                &self.data.subset(&cluster.assignment),
                self.config.num_tables,
            ) {
                Ok((puffinn_index, memory_used)) => {
                    self.puffinn_indices.push(Some(puffinn_index));
                    cluster.memory_used = memory_used;
                }
                Err(e) => {
                    error!(
                        "Failed to create Puffinn index for cluster {}: {:?}",
                        cluster_idx, e
                    );
                    return Err(ClusteredIndexError::PuffinnCreationError(e));
                }
            }
        }

        let indexing_duration = start_clustering.elapsed();

        info!(
            "Build process completed. Total clusters: {}, Indexing time: {:.2?}",
            total_clusters, indexing_duration
        );

        if let Some(metrics) = &mut self.metrics {
            metrics.log_index_building_time(indexing_duration);
        }

        Ok(())
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
    pub(crate) fn search(&mut self, query: &[T::DataType]) -> Result<Vec<(f32, usize)>> {
        if let Some(metrics) = &mut self.metrics {
            metrics.new_query();
            clear_distance_computations();
        }

        debug!(
            "Starting search procedure with parameters k={} and delta={:.2}",
            self.config.k, self.config.delta
        );
        let query_time = Instant::now();

        let delta_prime = self.config.delta;

        let sorted_cluster = self.sort_cluster_indices_by_distance(query);

        let mut priority_queue = TopKClosestHeap::new(self.config.k);

        let mut max_dist = f32::INFINITY;

        for cluster_idx in sorted_cluster {
            debug!("cluster index: {}", cluster_idx);
            let mut distance_computations = 0;
            let cluster_start = Instant::now();

            let cluster = &self.clusters[cluster_idx];

            // exit condition: if there are no more possible nearest neighbor stop
            // to see if there are no more possible nearest neighbor we check the top of the priority queue,
            // if the distance to the worst point in PQ is less than the distance of the nearest possible point in the cluster
            // then we can stop
            if let Some(top) = priority_queue.get_top() {
                debug!("top: {:?}", top);

                max_dist = top.1;

                // skips the first iteration so i dont have to worry about last_points being zero
                // log the distance computation of the exit condition
                distance_computations += 1;

                let cluster_min_distance =
                    self.data.distance_point(cluster.center_idx, query) - cluster.radius;
                if cluster_min_distance > top.1 {
                    if let Some(metrics) = &mut self.metrics {
                        metrics.add_distance_computation_cluster(distance_computations);
                        metrics.log_cluster_time(cluster_start.elapsed());
                    }

                    return Ok(priority_queue.to_list());
                }
            }

            let mut points_added = 0;
            if cluster.brute_force {
                // do brute force

                let candidates = self.brute_force_search(cluster, query)?;

                for (distance, p) in &candidates {
                    if priority_queue.add(Element {
                        distance: OrderedFloat(*distance),
                        point_index: *p,
                    }) {
                        points_added += 1;
                    }
                }

                distance_computations += candidates.len();
            } else {
                // do puffinn query algorithm

                let candidates = match &self.puffinn_indices[cluster.idx] {
                    Some(index) => index
                        .search::<T>(query, self.config.k, max_dist, delta_prime)
                        .map_err(ClusteredIndexError::PuffinnSearchError)?,
                    None => {
                        return Err(ClusteredIndexError::IndexNotFound());
                    }
                };

                // map puffinn result to the original dataset
                let mapped_candidates = match self.map_candidates(&candidates, cluster) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error on cluster {}", cluster_idx);
                        return Err(e);
                    }
                };

                let mut min_dist_cluster = f32::INFINITY;
                let mut max_dist_cluster = f32::NEG_INFINITY;
                for p in mapped_candidates {
                    let distance = self.data.distance_point(p, query);
                    if distance < min_dist_cluster {
                        min_dist_cluster = distance;
                    }
                    if distance > max_dist_cluster {
                        max_dist_cluster = distance;
                    }
                    if priority_queue.add(Element {
                        distance: OrderedFloat(distance),
                        point_index: p,
                    }) {
                        points_added += 1;
                    }
                }
                debug!(
                    "points_added = {}, min_dist = {}, max_dist = {}",
                    points_added, min_dist_cluster, max_dist_cluster
                );

                distance_computations += get_distance_computations() as usize;
            }

            debug!("Added {} points in cluster {})", points_added, cluster.idx);

            if let Some(metrics) = &mut self.metrics {
                metrics.log_n_candidates(points_added);
                metrics.log_cluster_time(cluster_start.elapsed());
                metrics.add_distance_computation_cluster(distance_computations);
            }
        }

        if let Some(metrics) = &mut self.metrics {
            metrics.log_query_time(query_time.elapsed());
        }

        Ok(priority_queue.to_list())
    }

    /// Saves metrics from a search run to a SQLite database.
    ///
    /// # Parameters
    /// - `db_path`: Path to SQLite database file
    /// - `granularity`: Level of detail for metrics (Run/Query/Cluster)
    /// - `ground_truth_distances`: True k-NN distances for computing recall
    /// - `run_distances`: Distances returned by the search algorithm
    /// - `total_search_time`: Total time spent on all queries
    ///
    /// # Errors
    /// - `ClusteredIndexError::MetricsError` if metrics are not enabled or database doesn't exist
    /// - `ClusteredIndexError::ResultDBError` for database connection/operation errors
    pub(crate) fn save_metrics(
        &mut self,
        db_path: String,
        granularity: MetricsGranularity,
        ground_truth_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
        total_search_time: &Duration,
    ) -> Result<()> {
        if !db_exists(&db_path) {
            return Err(ClusteredIndexError::MetricsError(format!(
                "No existing database in path {}",
                db_path
            )));
        }

        // Connect to the database
        let conn_res = Connection::open(db_path)
            .map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()));

        match conn_res {
            Ok(mut conn) => {
                if let Some(metrics) = &mut self.metrics {
                    return metrics.save_metrics(
                        &mut conn,
                        granularity,
                        &self.clusters,
                        ground_truth_distances,
                        run_distances,
                        total_search_time,
                    );
                } else {
                    return Err(ClusteredIndexError::MetricsError(
                        "run metrics are not enabled".to_string(),
                    ));
                }
            }
            Err(e) => return Err(e),
        }
    }

    /// Serializes the index to an HDF5 file.
    ///
    /// Saves:
    /// - Configuration parameters
    /// - Cluster information (centers, assignments, radii)
    /// - PUFFINN indices for each cluster
    ///
    /// # Parameters
    /// - `directory`: Directory where the index file will be saved
    ///
    /// # File naming
    /// The file is named: `index_{dataset_name}_k{clusters_factor}_L{num_tables}.h5`
    ///
    /// # Errors
    /// Returns `ClusteredIndexError::SerializeError` if:
    /// - Directory doesn't exist
    /// - File creation fails
    /// - Serialization of any component fails
    pub(crate) fn serialize(&self, directory: &str) -> Result<()> {
        if fs::metadata(directory).is_err() {
            return Err(ClusteredIndexError::SerializeError(format!(
                "directory {} doesn't exist",
                directory
            )));
        }

        let file_path = format!(
            "{}/index_{}_k{:.2}_L{}.h5",
            directory,
            self.config.dataset_name,
            self.config.num_clusters_factor,
            self.config.num_tables
        );
        let file = File::create(file_path.clone())
            .map_err(|e| ClusteredIndexError::SerializeError(e.to_string()))?;

        // write Config
        let config_json = serde_json::to_string(&self.config).unwrap();
        let config_ascii = VarLenAscii::from_ascii(&config_json).unwrap();
        file.new_dataset::<VarLenAscii>()
            .create("config")
            .unwrap()
            .write_scalar(&config_ascii)
            .map_err(|e| ClusteredIndexError::SerializeError(e.to_string()))?;

        // write all ClusterCenter
        let clusters_json = serde_json::to_string(&self.clusters).unwrap();
        let clusters_ascii = VarLenAscii::from_ascii(&clusters_json).unwrap();
        file.new_dataset::<VarLenUnicode>()
            .create("clusters")
            .unwrap()
            .write_scalar(&clusters_ascii)
            .map_err(|e| ClusteredIndexError::SerializeError(e.to_string()))?;

        // write all puffinn indexes
        for (index_id, puffinn_index) in self.puffinn_indices.iter().enumerate() {
            if let Some(index) = puffinn_index {
                index
                    .save_to_file(&file_path, index_id)
                    .map_err(ClusteredIndexError::SerializeError)?;
            }
        }

        Ok(())
    }

    /// Returns the total number of distance computations for the current query.
    ///
    /// # Returns
    /// Total number of distance computations if metrics are enabled
    ///
    /// # Errors
    /// Returns `ClusteredIndexError::MetricsError` if metrics are not enabled
    pub fn get_distance_computations(&self) -> Result<usize> {
        if let Some(metrics) = &self.metrics {
            return Ok(metrics.current_query().unwrap().distance_computations);
        }

        Err(ClusteredIndexError::MetricsError(
            "run metrics are not enabled".to_string(),
        ))
    }

    /// Sorts clusters by their distance from the query point.
    ///
    /// # Implementation
    /// 1. Computes distance from query to each cluster center
    /// 2. Sorts clusters by these distances in ascending order
    /// 3. Returns indices of clusters in sorted order
    ///
    /// This ordering is crucial for early termination and efficiency:
    /// - Closer clusters are more likely to contain nearest neighbors
    /// - Allows terminating search when minimum distance to next cluster exceeds current kth distance
    ///
    /// # Parameters
    /// - `query`: Query point to compute distances against
    ///
    /// # Returns
    /// Vector of cluster indices sorted by distance from query to cluster centers
    fn sort_cluster_indices_by_distance(&mut self, query: &[T::DataType]) -> Vec<usize> {
        let mut cluster_distances: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .map(|cluster| {
                let dist = self.data.distance_point(cluster.center_idx, query);
                (cluster.idx, dist)
            })
            .collect();

        // TODO: we can remove some distance computations from the main loop
        // since we compute each distance from the center to the query we dont actually
        // need to redo it in the exit condition
        if let Some(metrics) = &mut self.metrics {
            metrics.add_distance_computation_global(cluster_distances.len());
        }

        cluster_distances.sort_by(|&(_, dist_a), &(_, dist_b)| {
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        cluster_distances.into_iter().map(|(i, _)| i).collect()
    }

    /// Maps local indices from PUFFINN search results to global dataset indices.
    ///
    /// PUFFINN returns indices local to the subset of points in a cluster.
    /// This function maps them back to indices in the original dataset using
    /// the cluster's assignment vector.
    ///
    /// # Parameters
    /// - `candidates`: Vector of local indices from PUFFINN search
    /// - `cluster`: Cluster containing the mapping information
    ///
    /// # Returns
    /// Vector of global dataset indices corresponding to the local indices
    ///
    /// # Errors
    /// Returns `ClusteredIndexError::IndexOutOfBounds` if any local index
    /// exceeds the cluster's size
    fn map_candidates(&self, candidates: &[u32], cluster: &ClusterCenter) -> Result<Vec<usize>> {
        candidates
            .iter()
            .map(|&local_idx| {
                let local_idx = local_idx as usize;
                if local_idx < cluster.assignment.len() {
                    Ok(cluster.assignment[local_idx])
                } else {
                    Err(ClusteredIndexError::IndexOutOfBounds(
                        local_idx,
                        cluster.assignment.len(),
                    ))
                }
            })
            .collect::<Result<Vec<usize>>>()
    }

    /// Performs brute force search within a cluster.
    ///
    /// Used for small clusters where building an index would be inefficient.
    /// Computes distances to all points in the cluster and returns the k nearest.
    ///
    /// # Parameters
    /// - `cluster`: Cluster to search in
    /// - `query`: Query point
    ///
    /// # Returns
    /// Vector of (distance, index) pairs for the k nearest neighbors in the cluster,
    /// sorted by distance
    ///
    /// # Performance
    /// Time complexity: O(cluster_size * dim) where dim is point dimensionality
    fn brute_force_search(
        &self,
        cluster: &ClusterCenter,
        query: &[T::DataType],
    ) -> Result<Vec<(f32, usize)>> {
        let mut priority_queue = TopKClosestHeap::new(self.config.k);
        let mut points_added = 0;
        for p in &cluster.assignment {
            let distance = self.data.distance_point(*p, query);
            if priority_queue.add(Element {
                distance: OrderedFloat(distance),
                point_index: *p,
            }) {
                points_added += 1;
            }
        }

        debug!("points added in brute force: {}", points_added);
        Ok(priority_queue.to_list())
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::Config, metricdata::AngularData};
    use ndarray::arr2;

    use super::{ClusterCenter, ClusteredIndex};

    #[test]
    fn test_sort_cluster() {
        let points = arr2(&[
            [0.1, 0.9, 0.4],
            [0.7, 0.2, 0.6],
            [0.5, 0.3, 0.9],
            [0.8, 0.4, 0.1],
            [0.2, 0.1, 0.8],
            [0.9, 0.8, 0.3],
            [0.3, 0.6, 0.5],
            [0.4, 0.3, 0.7],
            [0.1, 0.2, 0.9],
            [0.6, 0.7, 0.8],
            [0.2, 0.8, 0.1],
            [0.9, 0.2, 0.4],
            [0.3, 0.5, 0.6],
            [0.1, 0.9, 0.2],
            [0.7, 0.4, 0.6],
            [0.8, 0.3, 0.2],
            [0.4, 0.6, 0.3],
            [0.2, 0.7, 0.9],
            [0.9, 0.4, 0.8],
            [0.5, 0.1, 0.3],
        ]);

        let data = AngularData::new(points);

        let cluster_indices: [usize; 3] = [6, 3, 17];
        let mut clusters = Vec::new();
        for (idx, center_idx) in cluster_indices.iter().enumerate() {
            clusters.push(ClusterCenter {
                idx,
                center_idx: *center_idx,
                radius: 0.0,
                assignment: vec![],
                brute_force: false,
                memory_used: 0,
            });
        }

        let config = Config::default();

        let mut index = ClusteredIndex {
            data,
            clusters,
            config,
            puffinn_indices: Vec::new(),
            metrics: None,
        };

        let sorted_indices = index.sort_cluster_indices_by_distance(&[0.1, 0.0, 0.7]);

        assert_eq!(sorted_indices, vec![2, 0, 1]);
    }
}
