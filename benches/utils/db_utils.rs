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

pub fn check_configuration_exists_clann(
    conn: &Connection,
    config: &Config,
    git_hash: &str,
) -> Result<bool, BenchmarkError> {
    let query = "
        SELECT created_at, recall_mean 
        FROM clann_results 
        WHERE num_clusters BETWEEN ?1 - 1e-6 AND ?1 + 1e-6 
        AND num_tables = ?2 
        AND k = ?3 
        AND delta BETWEEN ?4  - 1e-6 AND ?4 + 1e-6
        AND dataset = ?5 
        AND git_commit_hash = ?6
    ";

    let mut stmt = conn.prepare(query)?;
    let mut rows = stmt.query(params![
        config.num_clusters_factor,
        config.num_tables,
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
             Dataset: {}, Clusters: {}, L: {}, k: {}, delta: {}", 
            created_at,
            recall_mean,
            config.dataset_name,
            config.num_clusters_factor,
            config.num_tables,
            config.k,
            config.delta
        );
        
        Err(BenchmarkError::ConfigExists(msg))
    } else {
        Ok(false)
    }
}

pub fn check_configuration_exists_puffinn(
    conn: &Connection,
    config: &Config,
) -> Result<bool, rusqlite::Error> {
    let query = "
        SELECT created_at, queries_per_second 
        FROM puffinn_results 
        WHERE num_tables = ?1 
        AND k = ?2 
        AND delta BETWEEN ?3  - 1e-6 AND ?3 + 1e-6
        AND dataset = ?4
    ";

    let mut stmt = conn.prepare(query)?;
    let mut rows = stmt.query(params![
        config.num_tables,
        config.k,
        config.delta,
        config.dataset_name,
    ])?;

    if let Some(row) = rows.next()? {
        let created_at: String = row.get(0)?;
        let queries_per_second: f64 = row.get(1)?;
        
        println!(
            "Puffinn Configuration already tested on {} with {:.3} queries/sec:\n\
             Dataset: {}, L: {}, k: {}, delta: {}", 
            created_at,
            queries_per_second,
            config.dataset_name,
            config.num_tables,
            config.k,
            config.delta
        );
        
        Ok(true)
    } else {
        Ok(false)
    }
}