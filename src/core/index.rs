use ordered_float::OrderedFloat;
use tracing::{debug, info};

use crate::core::heap::Element;
use crate::core::{ClusteredIndexError, Config, Result};
use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::IndexableSimilarity;
use crate::puffinn_binds::PuffinnIndex;

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

        let k = config.k;

        Ok(ClusteredIndex {
            data,
            clusters: Vec::with_capacity(k),
            config,
            puffinn_indices: Vec::with_capacity(k),
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
        let (centers, assignment, radius) = greedy_minimum_maximum(&self.data, self.config.k);

        println!("centers: {}", centers.len());

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

    /// Search for the approximate k nearest neighbors to a query.
    ///
    /// # Parameters
    /// - `query`: A vector of the same type of the dataset representing the query point.
    /// - `k`: Number of neighbours to search for.
    /// - `delta`: Expected recall of the result.
    pub fn search(&self, query: &[T::DataType], k: usize, delta: f32) -> Result<Vec<(f32, usize)>> {
        info!("Starting search procedure with parameters k={} and delta={:.2}", k, delta);

        let delta_prime = 1.0 - (1.0 - delta) / (self.clusters.len() as f32);

        let sorted_cluster = self.sort_clusters_by_distance(query);

        let mut priority_queue = TopKClosestHeap::new(k);

        for cluster in sorted_cluster {
            println!("cluster");
            if let Some(top) = priority_queue.get_top() {
                if self.data.distance_point(cluster.center_idx, query) - cluster.radius
                    > self.data.distance_point(top, query)
                {
                    println!("Bye!");
                    return Ok(priority_queue.to_list());
                }
            }

            let candidates: Vec<u32> = self.puffinn_indices[cluster.idx]
                .search::<T>(query, k, delta_prime)
                .map_err(|e| ClusteredIndexError::PuffinnSearchError(e))?;

            let mapped_candidates: Vec<usize> = candidates
                .into_iter()
                .map(|local_idx| cluster.assignment[local_idx as usize])
                .collect();

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

            println!("Added {} points in cluster {}", points_added, cluster.idx);
        }

        Ok(priority_queue.to_list())
    }

    fn sort_clusters_by_distance(&self, query: &[T::DataType]) -> Vec<ClusterCenter> {
        let mut sorted_clusters = self.clusters.clone();
        sorted_clusters.sort_by(|a, b| {
            let dist_a = self.data.distance_point(a.center_idx, query);
            let dist_b = self.data.distance_point(b.center_idx, query);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_clusters
    }
}
