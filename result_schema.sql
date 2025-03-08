-- Build time metrics
CREATE TABLE build_metrics (
	num_clusters INTEGER NOT NULL, 
	num_tables INTEGER NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) DEFAULT 'NO_COMMIT' NOT NULL,
	dataset_len INTEGER,
	total_num_clusters INTEGER NOT NULL DEFAULT 0,
	greedy_num_clusters INTEGER NOT NULL DEFAULT 0,
	memory_used_bytes INTEGER, 
	build_time_s INTEGER,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_clusters, num_tables, dataset, git_commit_hash), 
	CONSTRAINT positive_clusters CHECK (num_clusters > 0), 
	CONSTRAINT positive_L CHECK (num_tables > 0) 
);

CREATE TABLE build_metrics_cluster (
	num_clusters INTEGER NOT NULL, 
	num_tables INTEGER NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) DEFAULT 'NO_COMMIT' NOT NULL,
	cluster_idx INTEGER NOT NULL,
	center_idx INTEGER,
	greedy_flag INTEGER,
	radius REAL,
	num_points INTEGER,
	memory_used_bytes INTEGER,
	PRIMARY KEY (num_clusters, num_tables, dataset, git_commit_hash, cluster_idx), 
	FOREIGN KEY (num_clusters, num_tables, dataset, git_commit_hash) REFERENCES build_metrics(num_clusters, num_tables, dataset, git_commit_hash) ON DELETE CASCADE
);

-- Search time metrics for all the queries
CREATE TABLE search_metrics ( 
	num_clusters INTEGER NOT NULL, 
	num_tables INTEGER NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) DEFAULT 'NO_COMMIT' NOT NULL, -- Using default instead of NULL,
	search_time_ms INTEGER, 
	queries_per_second REAL, 
	recall_mean REAL, 
	recall_std REAL, 
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_clusters, num_tables, k, delta, dataset, git_commit_hash), 
	FOREIGN KEY (num_clusters, num_tables, dataset, git_commit_hash) REFERENCES build_metrics(num_clusters, num_tables, dataset, git_commit_hash) ON DELETE CASCADE, 
	CONSTRAINT valid_recall CHECK (recall_mean >= 0 AND recall_mean <= 1), 
	CONSTRAINT valid_recall_std CHECK (recall_std >= 0), 
	CONSTRAINT positive_clusters CHECK (num_clusters > 0), 
	CONSTRAINT positive_k CHECK (k > 0), 
	CONSTRAINT positive_L CHECK (num_tables > 0) 
); 
	
-- Table for storing per-query metrics 
CREATE TABLE search_metrics_query (
	num_clusters INTEGER NOT NULL,
	num_tables INTEGER NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) NOT NULL, 
	query_idx INTEGER NOT NULL, 
	query_time_ms INTEGER, 
	distance_computations INTEGER,
	PRIMARY KEY (num_clusters, num_tables, k, delta, dataset, git_commit_hash, query_idx), 
	FOREIGN KEY (num_clusters, num_tables, k, delta, dataset, git_commit_hash) REFERENCES search_metrics(num_clusters, num_tables, k, delta, dataset, git_commit_hash) ON DELETE CASCADE, 
	CONSTRAINT positive_time CHECK (query_time_ms >= 0), 
	CONSTRAINT positive_computations CHECK (distance_computations >= 0) 
); 

-- Table for storing detailed per-cluster metrics for each query 
CREATE TABLE search_metrics_cluster ( 
	num_clusters INTEGER NOT NULL, 
	num_tables INTEGER NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	git_commit_hash CHAR(40) NOT NULL, 
	query_idx INTEGER NOT NULL, 
	cluster_idx INTEGER NOT NULL, 
	n_candidates INTEGER, 
	cluster_time_ms INTEGER, 
	cluster_distance_computations INTEGER, 
	PRIMARY KEY (num_clusters, num_tables, k, delta, dataset, git_commit_hash, query_idx, cluster_idx), 
	FOREIGN KEY (num_clusters, num_tables, k, delta, dataset, git_commit_hash, query_idx) REFERENCES search_metrics_query(num_clusters, num_tables, k, delta, dataset, git_commit_hash, query_idx) ON DELETE CASCADE, 
	CONSTRAINT positive_candidates CHECK (n_candidates >= 0), 
	CONSTRAINT positive_cluster_time CHECK (cluster_time_ms >= 0),
	CONSTRAINT positive_cluster_computations CHECK (cluster_distance_computations >= 0) 
);

-- Stores PUFFINN results to be compared against CLANN
CREATE TABLE puffinn_results ( 
	num_tables INTEGER NOT NULL, 
	k INTEGER NOT NULL, 
	delta REAL NOT NULL, 
	dataset TEXT NOT NULL, 
	dataset_len INTEGER,
	memory_used_bytes INTEGER, 
	total_time_ms INTEGER, 
	queries_per_second REAL, 
	recall_mean REAL, 
	recall_std REAL, 
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (num_tables, k, delta, dataset),
	CONSTRAINT valid_recall CHECK (recall_mean >= 0 AND recall_mean <= 1), 
	CONSTRAINT valid_recall_std CHECK (recall_std >= 0)
); 

CREATE TABLE puffinn_results_query (
    num_tables INTEGER NOT NULL,
    k INTEGER NOT NULL,
    delta REAL NOT NULL,
    dataset TEXT NOT NULL,
    query_idx INTEGER NOT NULL,
    query_time_ms INTEGER, 
    distance_computations INTEGER,
    PRIMARY KEY (num_tables, k, delta, dataset, query_idx),
    FOREIGN KEY (num_tables, k, delta, dataset) REFERENCES puffinn_results(num_tables, k, delta, dataset) ON DELETE CASCADE,
    CONSTRAINT positive_time CHECK (query_time_ms >= 0),
    CONSTRAINT positive_computations CHECK (distance_computations >= 0)
);