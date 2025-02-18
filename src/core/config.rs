use serde::{Deserialize, Serialize};

/// Parameters for the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Kb per point used by the index
    pub num_tables: usize,

    /// Factor that needs to be multiplied to sqrt(n)
    pub num_clusters_factor: f32,

    /// Number of nearest neighbors to search
    pub k: usize,

    /// Expected Recall
    pub delta: f32,

    /// Dataset name
    pub dataset_name: String,
}

impl Default for Config {
    fn default() -> Self {
        Self { 
            num_tables: 1,   
            num_clusters_factor: 1.0,
            k: 10, 
            delta: 0.9,
            dataset_name: "".to_string(),
        }
    }
}

impl Config {
    pub fn new(
        num_tables: usize,
        num_clusters_factor: f32,
        k: usize,
        delta: f32,
        dataset_name: &str,
    ) -> Self {
        Self{
            num_tables,
            num_clusters_factor,
            k,
            delta,
            dataset_name: dataset_name.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        
        // Check default values
        assert_eq!(config.num_tables, 1);
        assert_eq!(config.num_clusters_factor, 1.0);
    }

    #[test]
    fn test_new_config() {
        let config = Config::new(2048, 10.0, 10, 0.9, "");
        
        // Check custom values
        assert_eq!(config.num_tables, 2048);
        assert_eq!(config.num_clusters_factor, 10.0);
    }

    #[test]
    fn test_serialize_config() {
        let config = Config::new(2048, 10.0, 10, 0.9, "");
        
        // Check if it can serialize and deserialize
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&serialized).unwrap();
        
        // Assert the deserialized config matches the original
        assert_eq!(config.num_tables, deserialized.num_tables);
        assert_eq!(config.num_clusters_factor, deserialized.num_clusters_factor);
    }
}