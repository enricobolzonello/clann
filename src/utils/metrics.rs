use std::time::Duration;

use log::warn;
use ndarray::{Array, Ix1, Ix2};
use rusqlite::{params, Connection};

use crate::core::Config;

#[derive(Debug, Clone, Copy)]
pub enum MetricsGranularity {
    Run,             // Only overall run metrics
    Query,           // Run + per-query metrics
    Cluster,         // Run + per-query + per-cluster metrics
}

pub struct QueryMetrics {
    pub distance_computations: usize,     // Global distance computations
    pub query_time: Duration,
    pub cluster_n_candidates: Vec<usize>, // Number of candidates per cluster
    pub cluster_timings: Vec<Duration>,   // Timing for each cluster
    pub cluster_distance_computations: Vec<usize>, // Distance computations per cluster
}

pub struct RunMetrics {
    pub queries: Vec<QueryMetrics>, // Query metrics for the current run
    pub cluster_sizes: Vec<usize>,  // Sizes of the clusters
    config: Config,
    dataset_len: usize,
    greedy_clusters: usize,
    memory_used_bytes: usize,
    total_search_time_s: Duration,
    queries_per_second: f32,
    recall_mean: f32,
    recall_std: f32,
}

impl QueryMetrics {
    pub fn new() -> Self {
        Self {
            distance_computations: 0,
            query_time: Duration::default(),
            cluster_n_candidates: Vec::new(),
            cluster_timings: Vec::new(),
            cluster_distance_computations: Vec::new(),
        }
    }
}

impl RunMetrics {
    pub fn new(config: Config, dataset_len: usize) -> Self {
        Self {
            queries: Vec::new(),
            cluster_sizes: Vec::new(),
            config,
            memory_used_bytes: 0,
            total_search_time_s: Duration::default(),
            queries_per_second: 0.0,
            recall_mean: 0.0,
            recall_std: 0.0,
            dataset_len,
            greedy_clusters: 0,
        }
    }

    pub fn new_query(&mut self) {
        self.queries.push(QueryMetrics::new());
    }

    pub fn current_query_mut(&mut self) -> Option<&mut QueryMetrics> {
        self.queries.last_mut()
    }

    pub fn current_query(&self) -> Option<&QueryMetrics> {
        self.queries.iter().last()
    }

    pub fn log_n_candidates(&mut self, n_candidates: usize) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_n_candidates.push(n_candidates);
        }
    }

    pub fn log_cluster_time(&mut self, time: Duration) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_timings.push(time);
        }
    }

    pub fn log_cluster_size(&mut self, cluster_size: usize) {
        self.cluster_sizes.push(cluster_size);
    }

    pub fn log_query_time(&mut self, time: Duration) {
        if let Some(query) = self.current_query_mut() {
            query.query_time = time;
        }
    }

    pub fn add_greedy_cluster_count(&mut self) {
        self.greedy_clusters += 1;
    }

    pub fn add_distance_computation_global(&mut self, n_comp: usize) {
        if let Some(query) = self.current_query_mut() {
            query.distance_computations += n_comp;
        }
    }

    pub fn add_distance_computation_cluster(&mut self, n_comp: usize) {
        if let Some(query) = self.current_query_mut() {
            query.cluster_distance_computations.push(n_comp);
            query.distance_computations += n_comp;
        }
    }

    pub fn compute_run_statistics(
        &mut self,
        dataset_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
        dataset_len: usize,
        total_search_time: &Duration,
    ) {
        // Recall
        (self.recall_mean, self.recall_std) = self.compute_recall(dataset_distances, run_distances);

        // Memory used
        self.memory_used_bytes = dataset_len * self.config.kb_per_point * 1024;

        // Search time
        self.total_search_time_s = *total_search_time;

        // QPS
        self.queries_per_second = (run_distances.len() as f32) / (self.total_search_time_s.as_nanos() as f32 / 1_000_000_000.0);
    }

    fn compute_recall(
        &self,
        dataset_distances: &Array<f32, Ix2>,
        run_distances: &[Vec<f32>],
    ) -> (f32, f32) {
        let mut recalls = Vec::with_capacity(run_distances.len());

        for i in 0..run_distances.len() {
            // Get threshold from dataset (ground truth) distances
            let t = Self::threshold(&dataset_distances.row(i).to_owned(), self.config.k, 1e-3);

            // Count matches in our search results
            let mut actual = 0;
            for &d in run_distances[i].iter().take(self.config.k) {
                if d <= t {
                    actual += 1;
                }
            }
            recalls.push(actual as f32);
        }

        let mean_recall = recalls.iter().sum::<f32>() / (recalls.len() as f32 * self.config.k as f32);
        let std_recall = {
            let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
            (recalls.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / recalls.len() as f32).sqrt()
                / self.config.k as f32
        };

        (mean_recall, std_recall)
    }

    fn threshold(distances: &Array<f32, Ix1>, count: usize, epsilon: f32) -> f32 {
        // Assuming distances need to be sorted first since we're finding the k-th smallest
        let mut sorted_distances: Vec<f32> = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_distances[count - 1] + epsilon
    }

    pub fn save_to_sqlite(&self, connection: &mut Connection, granularity: MetricsGranularity) -> Result<(), rusqlite::Error> {
        // Start a transaction to ensure all inserts succeed or none do
        let tx = connection.transaction()?;

        // Always insert run-level metrics
        self.sqlite_insert_clann_results(&tx)?;

        // Insert query and cluster metrics based on granularity
        match granularity {
            MetricsGranularity::Run => (), // Only run metrics, already inserted
            MetricsGranularity::Query => {
                self.sqlite_insert_queries_only(&tx)?;
            }
            MetricsGranularity::Cluster => {
                self.sqlite_insert_clann_results_query(&tx)?;
            }
        }

        // Commit the transaction
        tx.commit()
    }

    fn sqlite_insert_clann_results(&self, conn: &Connection) -> Result<(), rusqlite::Error> {
        let current_time = chrono::Utc::now().to_rfc3339();
        let total_clusters = self.cluster_sizes.len();
    
        match conn.execute(
            "INSERT INTO clann_results (
                num_clusters,
                kb_per_point,
                k,
                delta,
                dataset,
                git_commit_hash,
                dataset_len,
                memory_used_bytes,
                total_time_ms,
                queries_per_second,
                recall_mean,
                recall_std,
                created_at,
                total_num_clusters,
                greedy_num_clusters
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                self.config.num_clusters_factor,
                self.config.kb_per_point,
                self.config.k,
                self.config.delta,
                self.config.dataset_name,
                option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT"),
                self.dataset_len,
                self.memory_used_bytes,
                self.total_search_time_s.as_secs_f32(),
                self.queries_per_second,
                self.recall_mean,
                self.recall_std,
                current_time,
                total_clusters,
                self.greedy_clusters,
            ],
        ) {
            Ok(_) => Ok(()),
            Err(e) => {
                if let rusqlite::Error::SqliteFailure(error, Some(message)) = &e {
                    if error.code == rusqlite::ErrorCode::ConstraintViolation
                        && message.contains("UNIQUE constraint failed")
                    {
                        warn!("Metrics not saved, results with this configuration already exist");
                        return Ok(());
                    }
                }
                Err(e)
            }
        }
    }

    fn sqlite_insert_queries_only(&self, conn: &Connection) -> Result<(), rusqlite::Error> {

        let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

        // Insert only query-level metrics
        for (query_idx, query) in self.queries.iter().enumerate() {
            self.sqlite_insert_query_metrics(conn, query_idx, query, &git_hash)?;
        }

        Ok(())
    }

    fn sqlite_insert_clann_results_query(&self, conn: &Connection) -> Result<(), rusqlite::Error> {

        let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

        // Insert query-level metrics
        for (query_idx, query) in self.queries.iter().enumerate() {
            self.sqlite_insert_query_metrics(conn, query_idx, query, &git_hash)?;
            
            // Insert cluster-level metrics for each query
            for (cluster_idx, ((n_candidates, timing), distance_comp)) in query
                .cluster_n_candidates
                .iter()
                .zip(&query.cluster_timings)
                .zip(&query.cluster_distance_computations)
                .enumerate()
            {
                self.sqlite_insert_cluster_metrics(
                    conn,
                    query_idx,
                    cluster_idx,
                    *n_candidates,
                    timing,
                    *distance_comp,
                    &git_hash,
                )?;
            }
        }

        Ok(())
    }

    fn sqlite_insert_query_metrics(
        &self,
        conn: &Connection,
        query_idx: usize,
        query: &QueryMetrics,
        git_hash: &str,
    ) -> Result<(), rusqlite::Error> {
        conn.execute(
            "INSERT INTO clann_results_query (
                num_clusters,
                kb_per_point,
                k,
                delta,
                dataset,
                git_commit_hash,
                query_idx,
                query_time_ms,
                distance_computations
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                self.config.num_clusters_factor,
                self.config.kb_per_point,
                self.config.k,
                self.config.delta,
                self.config.dataset_name,
                git_hash,
                query_idx as i64,
                query.query_time.as_millis() as i64,
                query.distance_computations as i64,
            ],
        )?;

        Ok(())
    }

    fn sqlite_insert_cluster_metrics(
        &self,
        conn: &Connection,
        query_idx: usize,
        cluster_idx: usize,
        n_candidates: usize,
        timing: &Duration,
        distance_comp: usize,
        git_hash: &str,
    ) -> Result<(), rusqlite::Error> {
        let cluster_size = self
            .cluster_sizes
            .get(cluster_idx)
            .unwrap_or(&0);

        conn.execute(
            "INSERT INTO clann_results_query_cluster (
                num_clusters,
                kb_per_point,
                k,
                delta,
                dataset,
                git_commit_hash,
                query_idx,
                cluster_idx,
                n_candidates,
                cluster_time_ms,
                cluster_size,
                cluster_distance_computations,
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                self.config.num_clusters_factor,
                self.config.kb_per_point,
                self.config.k,
                self.config.delta,
                self.config.dataset_name,
                git_hash,
                query_idx as i64,
                cluster_idx as i64,
                n_candidates as i64,
                timing.as_micros() as i64,
                *cluster_size as i64,
                distance_comp as i64,
            ],
        )?;

        Ok(())
    }
}
