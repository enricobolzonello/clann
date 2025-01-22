use clann::core::Config;
use rusqlite::{Connection, params};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("Rusqlite error")]
    Database(#[from] rusqlite::Error),

    #[error("Config already exists in DB")]
    ConfigExists(String),
}

pub fn check_configuration_exists(
    conn: &Connection,
    config: &Config,
    git_hash: &str,
) -> Result<bool, BenchmarkError> {
    let query = "
        SELECT created_at, recall_mean 
        FROM clann_results 
        WHERE num_clusters = ?1 
        AND kb_per_point = ?2 
        AND k = ?3 
        AND delta = ?4 
        AND dataset = ?5 
        AND git_commit_hash = ?6
    ";

    let mut stmt = conn.prepare(query)?;
    let mut rows = stmt.query(params![
        config.num_clusters_factor,
        config.kb_per_point,
        config.k,
        config.delta,
        config.dataset_name,
        git_hash,
    ])?;

    if let Some(row) = rows.next()? {
        let created_at: String = row.get(0)?;
        let recall_mean: f32 = row.get(1)?;
        
        let msg = format!(
            "Configuration already tested on {} with recall {:.3}:\n\
             Dataset: {}, Clusters: {}, KB/point: {}, k: {}, delta: {}", 
            created_at,
            recall_mean,
            config.dataset_name,
            config.num_clusters_factor,
            config.kb_per_point,
            config.k,
            config.delta
        );
        
        Err(BenchmarkError::ConfigExists(msg))
    } else {
        Ok(false)
    }
}