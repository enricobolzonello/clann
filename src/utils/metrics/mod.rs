use ndarray::{Array, Ix2};
use rusqlite::Connection;
use sqlite::{
    sqlite_build_metrics, sqlite_insert_clann_results, sqlite_insert_clann_results_query,
    sqlite_insert_queries_only,
};
use std::time::Duration;

use crate::core::{config::{MetricsGranularity, MetricsOutput}, index::ClusterCenter, ClusteredIndexError, Config};

use super::get_recall_values;
mod sqlite;

pub(crate) struct QueryMetrics {
    pub(crate) distance_computations: usize, // Global distance computations
    pub(crate) query_time: Duration,
    pub(crate) cluster_n_candidates: Vec<usize>, // Number of candidates per cluster
    pub(crate) cluster_timings: Vec<Duration>,   // Timing for each cluster
    pub(crate) cluster_distance_computations: Vec<usize>, // Distance computations per cluster
}

pub(crate) struct RunMetrics {
    // search metrics
    pub(crate) queries: Vec<QueryMetrics>,
    config: Config,
    dataset_len: usize,
    total_search_time_s: Duration,
    queries_per_second: f32,
    recall_mean: f32,
    recall_std: f32,

    // index metrics
    indexing_duration: Duration,
}

impl QueryMetrics {
    pub(crate) fn new() -> Self {
        Self {
            distance_computations: 0,
            query_time: Duration::default(),
            cluster_n_candidates: Vec::new(),
            cluster_timings: Vec::new(),
            cluster_distance_computations: Vec::new(),
        }
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self::new()
    }      
}

impl RunMetrics {
    pub(crate) fn new(config: Config, dataset_len: usize) -> Self {
        Self {
            queries: Vec::new(),
            config,
            total_search_time_s: Duration::ZERO,
            queries_per_second: 0.0,
            recall_mean: 0.0,
            recall_std: 0.0,
            dataset_len,
            indexing_duration: Duration::ZERO,
        }
    }

    pub(crate) fn new_query(&mut self) {
        self.queries.push(QueryMetrics::new());
    }

    pub(crate) fn current_query_mut(&mut self) -> Option<&mut QueryMetrics> {
        self.queries.last_mut()
    }

    pub(crate) fn current_query(&self) -> Option<&QueryMetrics> {
        self.queries.iter().last()
    }

    pub(crate) fn log_index_building_time(&mut self, time: Duration) {
        self.indexing_duration = time;
    }

    pub(crate) fn log_n_candidates(&mut self, n_candidates: usize) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_n_candidates.push(n_candidates);
        }
    }

    pub(crate) fn log_cluster_time(&mut self, time: Duration) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_timings.push(time);
        }
    }

    pub(crate) fn log_query_time(&mut self, time: Duration) {
        if let Some(query) = self.current_query_mut() {
            query.query_time = time;
        }
    }

    pub(crate) fn add_distance_computation_global(&mut self, n_comp: usize) {
        if let Some(query) = self.current_query_mut() {
            query.distance_computations += n_comp;
        }
    }

    pub(crate) fn add_distance_computation_cluster(&mut self, n_comp: usize) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_distance_computations.push(n_comp);
            query.distance_computations += n_comp;
        }
    }

    /// Save the results to the specified sqlite database, with the given granularity
    pub(crate) fn save_metrics(
        &mut self,
        connection: &mut Connection,
        granularity: MetricsGranularity,
        clusters: &Vec<ClusterCenter>,
        dataset_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
        total_search_time: &Duration,
    ) -> Result<(), ClusteredIndexError> {
        self.compute_run_statistics(
            dataset_distances, 
            run_distances, 
            total_search_time
        );

        // Start a transaction to ensure all inserts succeed or none do
        let tx = connection.transaction().map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()))?;

        // Always insert build and run-level metrics
        self.save_build_metrics(&tx, clusters)?;
        self.save_search_metrics(&tx)?;

        // Insert query and cluster metrics based on granularity
        match granularity {
            MetricsGranularity::Run => (), // Only run metrics, already inserted
            MetricsGranularity::Query => {
                self.save_search_metrics_query(&tx)?;
            }
            MetricsGranularity::Cluster => {
                self.save_search_metrics_cluster(&tx)?;
            }
        }

        tx.commit().map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()))
    }

    fn save_build_metrics(
        &self,
        conn: &Connection,
        clusters: &Vec<ClusterCenter>,
    ) -> Result<(), ClusteredIndexError> {
        let mut num_greedy = 0;
        let mut memory_used_bytes = 0;
        for cluster in clusters {
            if cluster.brute_force {
                num_greedy += 1;
            }

            memory_used_bytes += cluster.memory_used;
        }

        match self.config.metrics_output {
            MetricsOutput::DB => {
                return sqlite_build_metrics(
                    conn,
                    self.config.num_clusters_factor,
                    self.config.num_tables,
                    self.config.dataset_name.clone(),
                    self.dataset_len,
                    clusters,
                    num_greedy,
                    memory_used_bytes,
                    self.indexing_duration.as_secs(),
                ).map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()));
            }
            MetricsOutput::None => {} // do nothing
        }

        Ok(())
    }

    fn save_search_metrics(&self, conn: &Connection) -> Result<(), ClusteredIndexError> {
        match self.config.metrics_output {
            MetricsOutput::DB => {
                return sqlite_insert_clann_results(
                    conn,
                    self.config.num_clusters_factor,
                    self.config.num_tables,
                    self.config.k,
                    self.config.delta,
                    self.config.dataset_name.clone(),
                    self.total_search_time_s,
                    self.queries_per_second,
                    self.recall_mean,
                    self.recall_std,
                ).map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()))
            }
            MetricsOutput::None => {} // do nothing
        }

        Ok(())
    }

    fn save_search_metrics_query(&self, conn: &Connection) -> Result<(), ClusteredIndexError> {
        match self.config.metrics_output {
            MetricsOutput::DB => {
                return sqlite_insert_queries_only(
                    conn,
                    &self.queries,
                    self.config.num_clusters_factor,
                    self.config.num_tables,
                    self.config.k,
                    self.config.delta,
                    self.config.dataset_name.clone(),
                ).map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()))
            }
            MetricsOutput::None => {} // do nothing
        }

        Ok(())
    }

    fn save_search_metrics_cluster(&self, conn: &Connection) -> Result<(), ClusteredIndexError> {
        match self.config.metrics_output {
            MetricsOutput::DB => {
                return sqlite_insert_clann_results_query(
                    conn,
                    &self.queries,
                    self.config.num_clusters_factor,
                    self.config.num_tables,
                    self.config.k,
                    self.config.delta,
                    self.config.dataset_name.clone(),
                ).map_err(|e| ClusteredIndexError::ResultDBError(e.to_string()))
            }
            MetricsOutput::None => {} // do nothing
        }

        Ok(())
    }

    fn compute_run_statistics(
        &mut self,
        dataset_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
        total_search_time: &Duration,
    ) {
        // Recall
        (self.recall_mean, self.recall_std, _) =
            get_recall_values(dataset_distances, run_distances, run_distances.len());

        // Search time
        self.total_search_time_s = *total_search_time;

        // QPS
        self.queries_per_second = (run_distances.len() as f32)
            / (self.total_search_time_s.as_nanos() as f32 / 1_000_000_000.0);
    }
}
