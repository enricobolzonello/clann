use std::time::Instant;

use ordered_float::OrderedFloat;
use log::{debug, info};

use crate::core::heap::Element;
use crate::core::{ClusteredIndexError, Config, Result};
use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::IndexableSimilarity;
use crate::puffinn_binds::PuffinnIndex;
use crate::utils::metrics::RunMetrics;

use super::gmm::greedy_minimum_maximum;
use super::heap::TopKClosestHeap;

#[derive(Clone)]
struct ClusterCenter {
    pub idx: usize, // index of the cluster, corresponds to the index of the vec of puffinn indexes
    pub center_idx: usize, // index in the dataset for the center point
    pub radius: f32, // radius of the cluster
    pub assignment: Vec<usize>, // vector of indices in the dataset
}

pub struct ClusteredIndex<T>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    data: T,
    clusters: Vec<ClusterCenter>,
    config: Config,
    puffinn_indices: Vec<PuffinnIndex>,
    metrics: Option<RunMetrics>,
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
        let _ = config.validate().map_err(ClusteredIndexError::ConfigError);

        if data.num_points() == 0 {
            return Err(ClusteredIndexError::DataError("empty dataset".to_string()));
        }

        info!("Initializing Index with config {:?}", config);

        let k = config.num_clusters;

        Ok(ClusteredIndex {
            data,
            clusters: Vec::with_capacity(k),
            config,
            puffinn_indices: Vec::with_capacity(k),
            metrics: None
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
        // 1) PERFORM CLUSTERING
        let (centers, assignment, radius) = greedy_minimum_maximum(&self.data, self.config.num_clusters);

        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];

        for (data_idx, &center_pos) in assignment.iter().enumerate() {
            assignments[center_pos].push(data_idx);
        }

        self.clusters = centers
            .iter()
            .zip(radius.iter())
            .zip(assignments.into_iter())
            .enumerate()
            .map(|(idx, ((&center_idx, &radius), assignment_indexes))| ClusterCenter {
                idx,
                center_idx,
                radius,
                assignment: assignment_indexes,
            })
            .collect();

        // 2) CREATE PUFFINN INDEXES
        for cluster in &self.clusters {
            if cluster.assignment.is_empty() {
                debug!("Skipping empty cluster {}", cluster.center_idx);
                continue;
            }

            let cluster_memory_limit = ((cluster.assignment.len() as f64
                / self.data.num_points() as f64)
                * self.config.memory_limit as f64) as usize;

            debug!(
                "Cluster {} memory limit {}",
                cluster.center_idx, cluster_memory_limit
            );

            // TODO: i dont like the clone
            let index_data = self.data.subset(cluster.assignment.clone());
            let puffinn_index = PuffinnIndex::new(&index_data, cluster_memory_limit)
                .map_err(|e| ClusteredIndexError::PuffinnCreationError(e))?;

            info!("Cluster {} puffinn index built", cluster.center_idx);

            // Store the Puffinn index
            self.puffinn_indices.push(puffinn_index);
        }

        Ok(())
    }

    pub fn search_static(&mut self, query: &[T::DataType]) -> Result<Vec<(f32, usize)>> {
        if let (Some(k), Some(delta)) = (self.config.k, self.config.delta) {
            self.search(query, k, delta)
        }else{
            Err(ClusteredIndexError::ConfigError("This method can be called only if Config has been called with k and delta".to_string()))
        }
    }

    /// Search for the approximate k nearest neighbors to a query.
    ///
    /// # Parameters
    /// - `query`: A vector of the same type of the dataset representing the query point.
    /// - `k`: Number of neighbours to search for.
    /// - `delta`: Expected recall of the result.
    pub fn search(&mut self, query: &[T::DataType], k: usize, delta: f32) -> Result<Vec<(f32, usize)>> {
        let start_time = Instant::now();
        debug!("Starting search procedure with parameters k={} and delta={:.2}", k, delta);

        let delta_prime = 1.0 - (1.0 - delta) / (self.clusters.len() as f32);

        let sorted_cluster = self.sort_cluster_indices_by_distance(query);

        let mut priority_queue = TopKClosestHeap::new(k);

        let mut n_candidates = Vec::new();
        let mut cluster_timings = Vec::new();

        for cluster_idx in sorted_cluster {
            let cluster_start = Instant::now();

            let cluster= &self.clusters[cluster_idx];

            if let Some(top) = priority_queue.get_top() {
                if self.data.distance_point(cluster.center_idx, query) - cluster.radius
                    > self.data.distance_point(top, query)
                {
                    let total_duration = start_time.elapsed();
                    if let Some(metrics) = self.metrics.as_mut() {
                        metrics.log_query(n_candidates, cluster_timings, total_duration);
                    }
                    debug!("Search completed in {:?}", total_duration);
                    return Ok(priority_queue.to_list());
                }
            }

            let candidates: Vec<u32> = self.puffinn_indices[cluster.idx]
                .search::<T>(query, k, delta_prime)
                .map_err(|e| ClusteredIndexError::PuffinnSearchError(e))?;

            let mapped_candidates: Vec<usize> = self.map_candidates(&candidates, &cluster);

            let mut points_added = 0;
            for p in mapped_candidates {
                let distance = self.data.distance_point(p, query);
                if priority_queue.add(Element{
                    distance: OrderedFloat(distance),
                    point_index: p,
                }){
                    points_added += 1;
                }
            }

            let cluster_duration = cluster_start.elapsed();
            cluster_timings.push(cluster_duration);
            
            debug!("Added {} points in cluster {} (took {:?})", 
                points_added, cluster.idx, cluster_duration);
            n_candidates.push(points_added);
        }

        let total_duration = start_time.elapsed();
        if let Some(metrics) = self.metrics.as_mut() {
            metrics.log_query(n_candidates, cluster_timings, total_duration);
        }
        debug!("Search completed in {:?}", total_duration);

        Ok(priority_queue.to_list())
    }

    pub fn enable_metrics(&mut self) -> Result<()> {
        if let (Some(k), Some(delta)) = (self.config.k, self.config.delta) {
            self.metrics = Some(RunMetrics::new(
                self.config.num_clusters,
                k,
                delta,
                self.config.memory_limit,
            ));

            Ok(())
        }else{
            Err(ClusteredIndexError::ConfigError("Metrics can be enabled only with static k and delta".to_string()))
        }
    }

    pub fn save_metrics(&self, output_path: String) -> Result<()> {
        if let Some(metrics) = &self.metrics {
            metrics
                .save_to_csv(output_path)
                .map_err(|e| ClusteredIndexError::ConfigError(format!("Failed to save metrics: {}", e)))?;
        }
        Ok(())
    }
    
    fn sort_cluster_indices_by_distance(&self, query: &[T::DataType]) -> Vec<usize> {
        let mut cluster_distances: Vec<(usize, f32)> = self.clusters
            .iter()
            .enumerate()
            .map(|(i, cluster)| {
                let dist = self.data.distance_point(cluster.center_idx, query);
                (i, dist)
            })
            .collect();
    
        cluster_distances.sort_by(|&(_, dist_a), &(_, dist_b)| {
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });
    
        cluster_distances.into_iter().map(|(i, _)| i).collect()
    }

    fn map_candidates(&self, candidates: &Vec<u32>, cluster: &ClusterCenter) -> Vec<usize> {
        (*candidates)
            .iter()
            .map(|&local_idx| cluster.assignment[local_idx as usize])
            .collect()
    }
}
