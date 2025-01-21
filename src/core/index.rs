use std::time::Instant;

use ordered_float::OrderedFloat;
use log::{debug, info};

use crate::core::heap::Element;
use crate::core::{ClusteredIndexError, Config, Result};
use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::puffinn_index::get_distance_computations;
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

        let k = (config.num_clusters_factor as f64 * (data.num_points() as f64).sqrt()).floor() as usize;

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
        info!("num_clusters: {}", self.clusters.capacity());
        // 1) PERFORM CLUSTERING
        let (centers, assignment, radius) = greedy_minimum_maximum(&self.data, self.clusters.capacity());

        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];

        for (data_idx, &center_pos) in assignment.iter().enumerate() {
            assignments[center_pos].push(data_idx);
        }

        self.clusters = centers
            .iter()
            .zip(radius.iter())
            .zip(assignments)
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

            let cluster_memory_limit = cluster.assignment.len() * self.config.kb_per_point * 1024;

            info!(
                "Memory limit {}, points: {}",
                cluster_memory_limit, cluster.assignment.len()
            );

            // TODO: i dont like the clone
            let index_data = self.data.subset(cluster.assignment.clone());
            let puffinn_index = PuffinnIndex::new(&index_data, cluster_memory_limit)
                .map_err(ClusteredIndexError::PuffinnCreationError)?;

            info!("Cluster {} puffinn index built", cluster.center_idx);

            // Store the Puffinn index
            self.puffinn_indices.push(puffinn_index);
        }

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

        debug!("Starting search procedure with parameters k={} and delta={:.2}", self.config.k, self.config.delta);

        let delta_prime = 1.0 - (1.0 - self.config.delta) / (self.clusters.len() as f32);

        let sorted_cluster = self.sort_cluster_indices_by_distance(query);

        let mut priority_queue = TopKClosestHeap::new(self.config.k);

        for cluster_idx in sorted_cluster {
            let cluster_start = Instant::now();

            let cluster= &self.clusters[cluster_idx];

            if let Some(top) = priority_queue.get_top() {
    
                // log the distance computation
                if let Some(metrics) = &mut self.metrics {
                    metrics.add_distance_computation(1);
                }

                // TODO: this needs to be changed (distance computation can be avoided)
                if self.data.distance_point(cluster.center_idx, query) - cluster.radius
                    > top.1
                {
                    // log the distance computations of puffinn
                    if let Some(metrics) = &mut self.metrics {
                        metrics.add_distance_computation(get_distance_computations() as usize);
                    }

                    return Ok(priority_queue.to_list());
                }
            }

            let candidates: Vec<u32> = self.puffinn_indices[cluster.idx]
                .search::<T>(query, self.config.k, delta_prime)
                .map_err(ClusteredIndexError::PuffinnSearchError)?;

            let mapped_candidates: Vec<usize> = self.map_candidates(&candidates, cluster);

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
            debug!("Added {} points in cluster {} (took {:?})", 
                points_added, cluster.idx, cluster_duration);

            if let Some(metrics) = &mut self.metrics {
                metrics.log_n_candidates(points_added);
                metrics.log_cluster_time(cluster_duration);
            }
        }

        // log the distance computations of puffinn
        if let Some(metrics) = &mut self.metrics {
            metrics.add_distance_computation(get_distance_computations() as usize);
        }

        Ok(priority_queue.to_list())
    }

    pub fn enable_metrics(&mut self) -> Result<()> {

        self.metrics = Some(
            RunMetrics::new()
        );

        Ok(())
    }

    pub fn save_metrics(&self, output_path: String) -> Result<()> {
        if let Some(metrics) = &self.metrics {
            metrics
                .save_to_csv(output_path, &self.config)
                .map_err(|e| ClusteredIndexError::ConfigError(format!("Failed to save metrics: {}", e)))?;
        }
        Ok(())
    }
    
    fn sort_cluster_indices_by_distance(&mut self, query: &[T::DataType]) -> Vec<usize> {
        let mut cluster_distances: Vec<(usize, f32)> = self.clusters
            .iter()
            .enumerate()
            .map(|(i, cluster)| {
                let dist = self.data.distance_point(cluster.center_idx, query);
                (i, dist)
            })
            .collect();

        // TODO: we can remove some distance computations from the main loop
        // since we compute each distance from the center to the query we dont actually 
        // need to redo it in the exit condition
        if let Some(metrics) = &mut self.metrics {
            metrics.add_distance_computation(cluster_distances.len());
        }
    
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
