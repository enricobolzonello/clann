-- Main results table storing experiment configurations and overall metrics 
CREATE TABLE clann_results ( 
	num_clusters INTEGER NOT NULL, 
	kb_per_point REAL NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) DEFAULT 'NO_COMMIT' NOT NULL, -- Using default instead of NULL,
	dataset_len INTEGER,
	memory_used_bytes INTEGER, 
	total_time_ms INTEGER, 
	queries_per_second REAL, 
	recall_mean REAL, 
	recall_std REAL, 
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_clusters, kb_per_point, k, delta, dataset, git_commit_hash), 
	CONSTRAINT valid_recall CHECK (recall_mean >= 0 AND recall_mean <= 1), 
	CONSTRAINT valid_recall_std CHECK (recall_std >= 0), 
	CONSTRAINT positive_clusters CHECK (num_clusters > 0), 
	CONSTRAINT positive_k CHECK (k > 0), 
	CONSTRAINT positive_kb CHECK (kb_per_point > 0) 
); 
	
-- Table for storing per-query metrics 
CREATE TABLE clann_results_query (
	num_clusters INTEGER NOT NULL,
	kb_per_point REAL NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) NOT NULL, 
	query_idx INTEGER NOT NULL, 
	query_time_ms INTEGER, 
	distance_computations INTEGER, 
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_clusters, kb_per_point, k, delta, dataset, git_commit_hash, query_idx), 
	FOREIGN KEY (num_clusters, kb_per_point, k, delta, dataset, git_commit_hash) REFERENCES clann_results(num_clusters, kb_per_point, k, delta, dataset, git_commit_hash) ON DELETE CASCADE, 
	CONSTRAINT positive_time CHECK (query_time_ms >= 0), 
	CONSTRAINT positive_computations CHECK (distance_computations >= 0) ); 

-- Table for storing detailed per-cluster metrics for each query 
CREATE TABLE clann_results_query_cluster ( 
	num_clusters INTEGER NOT NULL, 
	kb_per_point REAL NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) NOT NULL, 
	query_idx INTEGER NOT NULL, 
	cluster_idx INTEGER NOT NULL, 
	n_candidates INTEGER, 
	cluster_time_ms INTEGER, 
	cluster_size INTEGER, 
	cluster_distance_computations INTEGER, 
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_clusters, kb_per_point, k, delta, dataset, git_commit_hash, query_idx, cluster_idx), 
	FOREIGN KEY (num_clusters, kb_per_point, k, delta, dataset, git_commit_hash, query_idx) REFERENCES clann_results_query(num_clusters, kb_per_point, k, delta, dataset, git_commit_hash, query_idx) ON DELETE CASCADE, 
	CONSTRAINT positive_candidates CHECK (n_candidates >= 0), 
	CONSTRAINT positive_cluster_time CHECK (cluster_time_ms >= 0),
	CONSTRAINT positive_cluster_size CHECK (cluster_size >= 0),
	CONSTRAINT positive_cluster_computations CHECK (cluster_distance_computations >= 0) );
