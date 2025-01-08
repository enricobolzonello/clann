use std::collections::HashMap;

use tracing::{debug, info};

use crate::core::{ClusteredIndexError, Config};
use crate::metricdata::MetricData;
use crate::puffinn_binds::IndexableSimilarity;
use crate::puffinn_binds::PuffinnIndex;

use super::gmm::greedy_minimum_maximum;

#[derive(Clone)]
struct ClusterCenter {
    pub index: usize,           // index in the dataset for the center point
    pub radius: f32,            // radius of the cluster
    pub assignment: Vec<usize>, // vector of indices in the dataset
}

pub struct ClusteredIndex<T: MetricData + IndexableSimilarity<T>> {
    data: T,
    clusters: Vec<ClusterCenter>,
    config: Config,
}

impl<T: MetricData + IndexableSimilarity<T>> ClusteredIndex<T> {
    /// Creates a new Clustered Index.
    ///
    /// # Parameters
    /// - `config`: Configuration object specifying clustering and memory constraints.
    /// - `data`: The dataset implementing `MetricData`.
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
    pub fn new(config: Config, data: T) -> Result<Self, ClusteredIndexError> {
        config
            .validate()
            .map_err(|e| ClusteredIndexError::ConfigError(e.to_string()));

        info!("Initializing Index with config {:?}", config);

        Ok(ClusteredIndex {
            data,
            clusters: Vec::with_capacity(config.k),
            config,
        })
    }

    pub fn build(&mut self) {

        // 1) PERFORM CLUSTERING
        let (centers, assignment, radius) = greedy_minimum_maximum(&self.data, self.config.k);

        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];

        // Create a mapping of center indices to their positions
        let center_to_pos: HashMap<usize, usize> = centers
            .iter()
            .enumerate()
            .map(|(pos, &center_idx)| (center_idx, pos))
            .collect();

        for (idx, &center_idx) in assignment.iter().enumerate() {
            if let Some(&pos) = center_to_pos.get(&center_idx) {
                assignments[pos].push(idx);
            }
        }

        self.clusters = centers
            .iter()
            .zip(radius.iter())
            .zip(assignments.into_iter())
            .map(
                |((&center_idx, &radius), assignment_indexes)| ClusterCenter {
                    index: center_idx,
                    radius,
                    assignment: assignment_indexes,
                },
            )
            .collect();


        // 2) CREATE PUFFINN INDEXES
        for cluster in &self.clusters {
            let cluster_memory_limit = ((cluster.assignment.len() as f64 / self.data.num_points() as f64) * self.config.memory_limit as f64) as usize;
        
            debug!("Cluster {} memory limit {}", cluster.index, cluster_memory_limit);

            let puffinn_index = PuffinnIndex::new(&self.data, cluster_memory_limit);

            for point_idx in &cluster.assignment {
                // TODO
                todo!()
            }
        }

    }

    pub fn search(&self, query_idx: usize, k: usize, delta: f32) -> Vec<(usize, f32)> {
        todo!()
    }
}
