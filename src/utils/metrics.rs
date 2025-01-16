use std::{fs::File, path::Path};
use std::time::Duration;
use chrono::Local;
use csv::Writer;

pub struct QueryMetrics {
    idx: usize,
    n_candidates: Vec<usize>,
    cluster_timings: Vec<Duration>,
    total_duration: Duration,
}

pub struct RunMetrics {
    // parameters
    num_clusters: usize,
    k: usize,
    delta: f32,
    total_memory: usize,

    // run data
    queries: Vec<QueryMetrics>,

    // helpers
    last_query_idx: usize,
}

impl QueryMetrics {
    pub fn new(
        idx: usize, 
        n_candidates: Vec<usize>,
        cluster_timings: Vec<Duration>,
        total_duration: Duration
    ) -> Self {
        Self { 
            idx, 
            n_candidates,
            cluster_timings,
            total_duration,
        }
    }
}

impl RunMetrics {
    pub fn new(num_clusters: usize, k: usize, delta: f32, total_memory: usize) -> Self {
        Self {
            num_clusters,
            k,
            delta,
            total_memory,
            queries: Vec::new(),
            last_query_idx: 0,
        }
    }

    pub fn log_query(
        &mut self, 
        n_candidates: Vec<usize>,
        cluster_timings: Vec<Duration>,
        total_duration: Duration
    ) {
        let query_metrics = QueryMetrics::new(
            self.last_query_idx,
            n_candidates,
            cluster_timings,
            total_duration
        );

        self.queries.push(query_metrics);
        self.last_query_idx += 1;
    }

    pub fn save_to_csv(&self, output_path: String) -> Result<(), Box<dyn std::error::Error>> {
        if self.queries.is_empty() {
            return Err("There is no query data to save".into());
        }

        // Ensure the directory exists
        let output_dir = Path::new(&output_path);
        if !output_dir.exists() {
            return Err(format!("Output directory {} does not exist", output_path).into());
        }

        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filepath = format!("{}/metrics_{}.csv", output_path, timestamp);
        let file = File::create(&filepath)
            .map_err(|e| format!("Failed to create file {}: {}", filepath, e))?;
        let mut writer = Writer::from_writer(file);

        // Write parameters as metadata
        writer
            .write_record(&["param", "total_clusters", &self.num_clusters.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(&["param", "K", &self.k.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(&["param", "delta", &self.delta.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(&["param", "total_memory", &self.total_memory.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;

        // Write header
        writer
            .write_record(&[
                "query",
                "cluster_idx",
                "n_candidates",
                "cluster_duration_ms",
                "total_duration_ms"
            ])
            .map_err(|e| format!("Failed to write header to {}: {}", filepath, e))?;

        // Write query data
        for query in &self.queries {
            for idx in 0..query.n_candidates.len() {
                writer
                    .write_record(&[
                        query.idx.to_string(),
                        idx.to_string(),
                        query.n_candidates[idx].to_string(),
                        query.cluster_timings[idx].as_nanos().to_string(),
                        query.total_duration.as_nanos().to_string(),
                    ])
                    .map_err(|e| format!("Failed to write query data to {}: {}", filepath, e))?;
            }
        }

        // Flush writer
        writer
            .flush()
            .map_err(|e| format!("Failed to flush writer for {}: {}", filepath, e))?;

        Ok(())
    }
}