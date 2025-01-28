use std::time::{Duration, Instant};

use log::{debug, error, info, trace, warn};
use ndarray::{Array, Ix2};
use ordered_float::OrderedFloat;
use rusqlite::Connection;

use crate::core::heap::Element;
use crate::core::{ClusteredIndexError, Config, Result};
use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::get_distance_computations;
use crate::puffinn_binds::IndexableSimilarity;
use crate::puffinn_binds::PuffinnIndex;
use crate::utils::db_exists;
use crate::utils::{MetricsGranularity, RunMetrics};

use super::gmm::greedy_minimum_maximum;
use super::heap::TopKClosestHeap;

#[derive(Clone)]
struct ClusterCenter {
    pub idx: usize,             // index of the cluster, corresponds to the index of the vec of puffinn indexes
    pub center_idx: usize,      // index of the center point in the original dataset
    pub radius: f32,            // radius of the cluster
    pub assignment: Vec<usize>, // vector of indices to the original dataset for points assigned to this cluster
    pub brute_force: bool,      // flag indicating if brute force is applied instead of puffinn (<500 points)
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
    pub metrics: Option<RunMetrics>,
}

impl<T> ClusteredIndex<T>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    /// Creates a new Clustered Index.
    ///
    /// # Parameters
    /// - `config`: Configuration object specifying clustering and memory constraints.
    /// - `data`: The dataset implementing `MetricData`, `Subset` and `IndexableSimilarity`.
    ///
    /// # Errors
    /// Returns a `ClusteredIndexError::ConfigError` if the configuration is invalid.
    ///
    /// # Examples
    /// ```
    /// let config = Config::default();
    /// let data = EuclideanData::new(your_array);
    /// let index = ClusteredIndex::new(config, data).unwrap();
    /// ```
    pub fn new(config: Config, data: T) -> Result<Self> {
        if data.num_points() == 0 {
            return Err(ClusteredIndexError::DataError("empty dataset".to_string()));
        }

        info!("Initializing Index with config {:?}", config);

        let k = (config.num_clusters_factor as f64 * (data.num_points() as f64).sqrt()).floor()
            as usize;

        Ok(ClusteredIndex {
            data,
            clusters: Vec::with_capacity(k),
            config,
            puffinn_indices: Vec::with_capacity(k),
            metrics: None,
        })
    }

    /// Builds the index with the provided config and data
    ///
    /// # Errors
    /// Returns a `ClusteredIndexError::PuffinnCreationError` if there are any errors in one of the PUFFINN index creation
    ///
    /// # Examples
    /// ```
    /// index.build();
    /// ```
    pub fn build(&mut self) -> Result<()> {
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
                    brute_force: assignment_indexes.len() < 500,
                    assignment: assignment_indexes,
                };

                trace!(
                    "Cluster {}: center_idx={}, points={}, radius={}, brute force? {}",
                    idx,
                    cluster.center_idx,
                    cluster.assignment.len(),
                    cluster.radius,
                    cluster.brute_force
                );

                cluster
            })
            .collect();

        // 2) CREATE PUFFINN INDEXES
        info!("Creating Puffinn indexes...");
        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            if cluster.assignment.is_empty() {
                debug!("Skipping empty cluster {}", cluster_idx);
                continue;
            }

            if cluster.brute_force {
                self.puffinn_indices.push(None);
                info!("Skipping cluster {}, will do brute force", cluster_idx);
                if let Some(metrics) = &mut self.metrics {
                    metrics.add_greedy_cluster_count();
                }
                continue;
            }

            if let Some(metrics) = &mut self.metrics {
                metrics.log_cluster_size(cluster.assignment.len());
            }

            let cluster_memory_limit = cluster.assignment.len() * self.config.kb_per_point * 1024;

            debug!(
                "Cluster {}: Memory limit {}, points: {}",
                cluster_idx,
                cluster_memory_limit,
                cluster.assignment.len()
            );

            // Progress logging
            if cluster_idx % 10 == 0 {
                info!(
                    "Processing cluster {}/{} ({}%)",
                    cluster_idx + 1,
                    total_clusters,
                    ((cluster_idx + 1) as f32 / total_clusters as f32 * 100.0).round()
                );
            }

            // Create Puffinn index
            // TODO: i dont like the clone
            match PuffinnIndex::new(
                &self.data.subset(cluster.assignment.clone()),
                cluster_memory_limit,
            ) {
                Ok(puffinn_index) => {
                    self.puffinn_indices.push(Some(puffinn_index));
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

        Ok(())
    }

    /// Search for the approximate k nearest neighbors to a query.
    ///
    /// # Parameters
    /// - `query`: A vector of the same type of the dataset representing the query point.
    pub fn search(&mut self, query: &[T::DataType]) -> Result<Vec<(f32, usize)>> {
        if let Some(metrics) = &mut self.metrics {
            metrics.new_query();
        }

        debug!(
            "Starting search procedure with parameters k={} and delta={:.2}",
            self.config.k, self.config.delta
        );

        let delta_prime = 1.0 - (1.0 - self.config.delta) / (self.clusters.len() as f32);

        let sorted_cluster = self.sort_cluster_indices_by_distance(query);

        let mut priority_queue = TopKClosestHeap::new(self.config.k);

        for cluster_idx in sorted_cluster {
            let mut distance_computations = 0;
            let cluster_start = Instant::now();

            let cluster = &self.clusters[cluster_idx];

            if let Some(top) = priority_queue.get_top() {
                // log the distance computation of the exit condition
                distance_computations += 1;

                // TODO
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
                    Some(index) => {
                        index.search::<T>(query, self.config.k, delta_prime).map_err(ClusteredIndexError::PuffinnSearchError)?
                    },
                    None => {
                        return Err(ClusteredIndexError::IndexNotFound());
                    }
                };

                // map puffinn result to the original dataset
                let mapped_candidates: Vec<usize> = self.map_candidates(&candidates, cluster)?;

                for p in mapped_candidates {
                    let distance = self.data.distance_point(p, query);
                    if priority_queue.add(Element {
                        distance: OrderedFloat(distance),
                        point_index: p,
                    }) {
                        points_added += 1;
                    }
                }

                distance_computations += get_distance_computations() as usize;
            }

            debug!("Added {} points in cluster {})", points_added, cluster.idx);

            if let Some(metrics) = &mut self.metrics {
                metrics.log_n_candidates(points_added);
                metrics.log_cluster_time(cluster_start.elapsed());
                metrics.add_distance_computation_cluster(distance_computations);
            }
        }

        Ok(priority_queue.to_list())
    }

    /// Enables run metrics collection
    pub fn enable_metrics(&mut self) -> Result<()> {
        self.metrics = Some(RunMetrics::new(self.config.clone(), self.data.num_points()));

        Ok(())
    }

    /// Saves metrics to the specified sqlite3 database with the desired granularity. For example, if you select [`MetricsGranularity::Run`] only metrics for the whole run, like recall or total search time, are saved.
    /// 
    /// # Parameters
    /// - `db_path`: Path to the sqlite3 database
    /// - `granularity`: [`MetricsGranularity`] to specify which metrics need to be saved
    /// - `ground_truth_distances`: Ground truth distances to calculate recall
    /// - `run_distances`: Final distances returned by the search algorithm for all queries
    /// - `dataset_len`: Length of the train dataset
    /// - `total_search_time`: Search time for all queries
    pub fn save_metrics(
        &mut self,
        db_path: String,
        granularity: MetricsGranularity,
        ground_truth_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
        dataset_len: usize,
        total_search_time: &Duration,
    ) -> Result<()> {
        if !db_exists(&db_path) {
            eprintln!("No existing database in path {}", db_path);
        }

        // Connect to the database
        let conn_res = Connection::open(db_path)
            .map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()));

        match conn_res {
            Ok(mut conn) => {
                if let Some(metrics) = &mut self.metrics {
                    metrics.compute_run_statistics(
                        ground_truth_distances,
                        run_distances,
                        dataset_len,
                        total_search_time,
                    );

                    return metrics
                        .save_to_sqlite(&mut conn, granularity)
                        .map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()));
                } else {
                    warn!("Metrics not enabled!");
                }
            }
            Err(e) => return Err(e),
        }

        Ok(())
    }

    /// Helper to sort clusters in ascending order by distance from the query point
    /// Returns only the indices
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

    fn map_candidates(&self, candidates: &[u32], cluster: &ClusterCenter) -> Result<Vec<usize>> {
        let mapped: Vec<usize> = candidates
            .iter()
            .map(|&local_idx| {
                (local_idx as usize)
                    .try_into()
                    .map_err(|_| ClusteredIndexError::IndexMappingError(local_idx))
            })
            .filter_map(|res| res.ok())
            .map(|local_idx| {
                if local_idx < cluster.assignment.len() {
                    Ok(cluster.assignment[local_idx])
                } else {
                    Err(ClusteredIndexError::IndexOutOfBounds(local_idx,cluster.assignment.len()))
                }
            })
            .collect::<Result<Vec<usize>>>()?;
    
        Ok(mapped)
    }

    // Simple brute force search for small clusters (under 500 points)
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
