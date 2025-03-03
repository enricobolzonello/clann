pub fn csv_build_metrics(
    &self, 
    file_path: &str,
    num_clusters: usize,
    num_greedy: usize,
    memory_used_bytes: usize,
    build_times_s: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(file_path)?;

    wtr.write_record(&["num_clusters", &self.config.num_clusters_factor.to_string()])?;
    wtr.write_record(&["num_tables", &self.config.num_tables.to_string()])?;
    wtr.write_record(&["dataset", &self.config.dataset_name])?;
    wtr.write_record(&["total_num_clusters", &num_clusters.to_string()])?;
    wtr.write_record(&["greedy_num_clusters", &num_greedy.to_string()])?;
    wtr.write_record(&["memory_used_bytes", &memory_used_bytes.to_string()])?;
    wtr.write_record(&["build_time_s", &build_times_s.to_string()])?;

    Ok(())
}