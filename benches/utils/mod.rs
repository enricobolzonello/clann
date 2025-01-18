use clann::core::Config;

pub const CONFIGS: &[Config] = &[
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024, // 1GB
        num_clusters: 4,
        k: 10,
        delta: 0.9,
    },
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024,
        num_clusters: 8,
        k: 10,
        delta: 0.9,
    },
    Config {
        memory_limit: 1 * 1024 * 1024 * 1024, 
        num_clusters: 12,
        k: 10,
        delta: 0.9,
    },
];