use std::time::Duration;

use log::warn;
use rusqlite::{params, Connection};

use crate::core::index::ClusterCenter;

use super::QueryMetrics;

pub(crate) fn sqlite_build_metrics(
    conn: &Connection,
    num_clusters_factor: f32,
    num_tables: usize,
    dataset_name: String,
    dataset_len: usize,
    clusters: &Vec<ClusterCenter>,
    num_greedy: usize,
    memory_used_bytes: usize,
    build_times_s: u64,
) -> Result<(), rusqlite::Error> {
    let current_time = chrono::Utc::now().to_rfc3339();

    match conn.execute(
        "INSERT INTO build_metrics (
            num_clusters,
            num_tables,
            dataset,
            git_commit_hash,
            dataset_len,
            total_num_clusters,
            greedy_num_clusters,
            memory_used_bytes,
            build_time_s,
            created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            num_clusters_factor,
            num_tables,
            dataset_name,
            option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT"),
            dataset_len,
            clusters.len(),
            num_greedy,
            memory_used_bytes,
            build_times_s,
            current_time
        ],
    ) {
        Ok(_) => {},
        Err(e) => {
            if let rusqlite::Error::SqliteFailure(error, Some(message)) = &e {
                if error.code == rusqlite::ErrorCode::ConstraintViolation
                    && message.contains("UNIQUE constraint failed")
                {
                    warn!("Build metrics for this index already exist");
                    return Ok(());
                }
            }
            return Err(e);
        }
    };

    for cluster in clusters {
        match conn.execute(
            "INSERT INTO build_metrics_cluster (
                num_clusters,
                num_tables,
                dataset,
                git_commit_hash,
                cluster_idx,
                center_idx,
                greedy_flag,
                radius,
                num_points,
                memory_used_bytes
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                num_clusters_factor,
                num_tables,
                dataset_name,
                option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT"),
                cluster.idx,
                cluster.center_idx,
                if cluster.brute_force { 1 } else { 0 },
                cluster.radius,
                cluster.assignment.len(),
                cluster.memory_used,
            ],
        ) {
            Ok(_) => {},
            Err(e) => {
                if let rusqlite::Error::SqliteFailure(error, Some(message)) = &e {
                    if error.code == rusqlite::ErrorCode::ConstraintViolation
                        && message.contains("UNIQUE constraint failed")
                    {
                        warn!("Build metrics for this index already exist");
                        return Ok(());
                    }
                }
                return Err(e);
            }
        };
    }

    Ok(())
}

pub(crate) fn sqlite_insert_clann_results(
    conn: &Connection,
    num_clusters_factor: f32,
    num_tables: usize,
    k: usize,
    delta: f32,
    dataset_name: String,
    total_search_time_s: Duration,
    queries_per_second: f32,
    recall_mean: f32,
    recall_std: f32
) -> Result<(), rusqlite::Error> {
    let current_time = chrono::Utc::now().to_rfc3339();

    match conn.execute(
        "INSERT INTO search_metrics (
            num_clusters,
            num_tables,
            k,
            delta,
            dataset,
            git_commit_hash,
            total_time_ms,
            queries_per_second,
            recall_mean,
            recall_std,
            created_at,
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            num_clusters_factor,
            num_tables,
            k,
            delta,
            dataset_name,
            option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT"),
            total_search_time_s.as_secs_f32(),
            queries_per_second,
            recall_mean,
            recall_std,
            current_time,
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

pub(crate) fn sqlite_insert_queries_only(
    conn: &Connection,
    queries: &[QueryMetrics],
    num_clusters_factor: f32,
    num_tables: usize,
    k: usize,
    delta: f32,
    dataset_name: String,
) -> Result<(), rusqlite::Error> {

    let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

    // Insert only query-level metrics
    for (query_idx, query) in queries.iter().enumerate() {
        conn.execute(
            "INSERT INTO search_metrics_query (
                num_clusters,
                num_tables,
                k,
                delta,
                dataset,
                git_commit_hash,
                query_idx,
                query_time_ms,
                distance_computations
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                num_clusters_factor,
                num_tables,
                k,
                delta,
                dataset_name,
                git_hash,
                query_idx as i64,
                query.query_time.as_millis() as i64,
                query.distance_computations as i64,
            ],
        )?;
    }

    Ok(())
}

pub(crate) fn sqlite_insert_clann_results_query(
    conn: &Connection,
    queries: &[QueryMetrics],
    num_clusters_factor: f32,
    num_tables: usize,
    k: usize,
    delta: f32,
    dataset_name: String,
) -> Result<(), rusqlite::Error> {

    let git_hash = option_env!("GIT_COMMIT_HASH").unwrap_or("NO_COMMIT");

    // Insert query-level metrics
    for (query_idx, query) in queries.iter().enumerate() {
        conn.execute(
            "INSERT INTO search_metrics_query (
                num_clusters,
                num_tables,
                k,
                delta,
                dataset,
                git_commit_hash,
                query_idx,
                query_time_ms,
                distance_computations
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                num_clusters_factor,
                num_tables,
                k,
                delta,
                dataset_name,
                git_hash,
                query_idx as i64,
                query.query_time.as_millis() as i64,
                query.distance_computations as i64,
            ],
        )?;
        
        // Insert cluster-level metrics for each query
        for (cluster_idx, ((n_candidates, timing), distance_comp)) in query
            .cluster_n_candidates
            .iter()
            .zip(&query.cluster_timings)
            .zip(&query.cluster_distance_computations)
            .enumerate()
        {
            conn.execute(
                "INSERT INTO search_metrics_cluster (
                    num_clusters,
                    num_tables,
                    k,
                    delta,
                    dataset,
                    git_commit_hash,
                    query_idx,
                    cluster_idx,
                    n_candidates,
                    cluster_time_ms,
                    cluster_distance_computations
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    num_clusters_factor,
                    num_tables,
                    k,
                    delta,
                    dataset_name,
                    git_hash,
                    query_idx as i64,
                    cluster_idx as i64,
                    *n_candidates as i64,
                    timing.as_micros() as i64,
                    *distance_comp as i64,
                ],
            )?;
        }
    }

    Ok(())
}