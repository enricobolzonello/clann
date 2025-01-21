use std::{fs::File, path::Path};
use std::time::Duration;
use chrono::Local;
use csv::Writer;

use crate::core::Config;

pub struct QueryMetrics {
    n_candidates: Vec<usize>,
    cluster_timings: Vec<Duration>,
    pub distance_computations: usize,
    // TODO
    // cluster size
    // cluster distance computations
}

pub struct RunMetrics {
    // run data
    pub queries: Vec<QueryMetrics>,
}

impl QueryMetrics {
    pub fn new(
    ) -> Self {
        Self { 
            n_candidates: Vec::new(),
            cluster_timings: Vec::new(),
            distance_computations: 0
        }
    }
}

impl RunMetrics {
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    pub fn new_query(&mut self) {
        self.queries.push(QueryMetrics::new());
    }

    pub fn current_query_mut(&mut self) -> &mut QueryMetrics {
        let n = self.queries.len();
        &mut self.queries[n - 1]
    }

    pub fn current_query(&self) -> &QueryMetrics {
        let n = self.queries.len();
        &self.queries[n - 1]
    }

    pub fn log_n_candidates(&mut self, n_candidates: usize) {
        self.current_query_mut().n_candidates.push(n_candidates);
    }

    pub fn log_cluster_time(&mut self, time: Duration) {
        self.current_query_mut().cluster_timings.push(time);
    }

    pub fn add_distance_computation(&mut self, n_comp: usize) {
        self.current_query_mut().distance_computations += n_comp;
    }

    pub fn save_to_csv(&self, output_path: String, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
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
            .write_record(["param", "cluster_factor", &config.num_clusters_factor.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(["param", "K", &config.k.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(["param", "delta", &config.delta.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;
        writer
            .write_record(["param", "kb_per_point", &config.kb_per_point.to_string(), "-1", "-1"])
            .map_err(|e| format!("Failed to write metadata to {}: {}", filepath, e))?;

        // Write header
        writer
            .write_record([
                "query",
                "cluster_idx",
                "n_candidates",
                "cluster_duration_μs",
                "total_distances_computed"
            ])
            .map_err(|e| format!("Failed to write header to {}: {}", filepath, e))?;

        // Write query data
        for (query_idx, query) in (&self.queries).iter().enumerate() {
            for idx in 0..query.n_candidates.len() {
                writer
                    .write_record([
                        query_idx.to_string(),
                        idx.to_string(),
                        query.n_candidates[idx].to_string(),
                        query.cluster_timings[idx].as_micros().to_string(),
                        query.distance_computations.to_string(),
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

    pub fn print_summary(&self) {
        println!("\nMetrics Summary:");
        println!("Total queries: {}", self.queries.len());
        
        if !self.queries.is_empty() {
            let total_computations: usize = self.queries
                .iter()
                .map(|q| q.distance_computations)
                .sum();
            
            let total_time: f64 = self.queries
                .iter()
                .flat_map(|q| q.cluster_timings.iter())
                .sum::<Duration>()
                .as_secs_f64() * 1000.0;

            println!("Total distance computations: {}", total_computations);
            println!("Average computations per query: {:.2}", 
                total_computations as f64 / self.queries.len() as f64);
            println!("Total clustering time: {:.2}ms", total_time);
            println!("Average time per query: {:.2}ms", 
                total_time / self.queries.len() as f64);
        }
    }
}