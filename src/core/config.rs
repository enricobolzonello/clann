use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsOutput{
    DB,
    None
}

pub enum MetricsGranularity {
    Run,     // Only overall run metrics
    Query,   // Run + per-query metrics
    Cluster, // Run + per-query + per-cluster metrics
}

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

    // Where to save metrics
    pub metrics_output: MetricsOutput,
}

impl Default for Config {
    fn default() -> Self {
        Self { 
            num_tables: 10,   
            num_clusters_factor: 1.0,
            k: 10, 
            delta: 0.9,
            dataset_name: "".to_string(),
            metrics_output: MetricsOutput::None
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
        metrics_output: MetricsOutput
    ) -> Self {
        Self{
            num_tables,
            num_clusters_factor,
            k,
            delta,
            dataset_name: dataset_name.to_string(),
            metrics_output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        // Check all default values
        assert_eq!(config.num_tables, 1);
        assert_eq!(config.num_clusters_factor, 1.0);
        assert_eq!(config.k, 10);
        assert_eq!(config.delta, 0.9);
        assert_eq!(config.dataset_name, "");
        assert!(matches!(config.metrics_output, MetricsOutput::None));
    }

    #[test]
    fn test_new_config() {
        let metrics_output = MetricsOutput::None;
        let dataset_name = "test_dataset";
        let config = Config::new(2048, 10.0, 100, 0.95, dataset_name, metrics_output);
        
        // Check all custom values
        assert_eq!(config.num_tables, 2048);
        assert_eq!(config.num_clusters_factor, 10.0);
        assert_eq!(config.k, 100);
        assert_eq!(config.delta, 0.95);
        assert_eq!(config.dataset_name, dataset_name);
        assert!(matches!(config.metrics_output, MetricsOutput::None));
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let original = Config::new(
            2048, 
            10.0, 
            100, 
            0.95, 
            "test_dataset", 
            MetricsOutput::None
        );
        
        // Serialize to JSON
        let serialized = serde_json::to_string(&original).unwrap();
        
        // Verify JSON contains expected fields
        assert!(serialized.contains("\"num_tables\":2048"));
        assert!(serialized.contains("\"num_clusters_factor\":10.0"));
        assert!(serialized.contains("\"k\":100"));
        assert!(serialized.contains("\"delta\":0.95"));
        assert!(serialized.contains("\"dataset_name\":\"test_dataset\""));
        
        // Deserialize back to Config
        let deserialized: Config = serde_json::from_str(&serialized).unwrap();
        
        // Assert all fields match
        assert_eq!(original.num_tables, deserialized.num_tables);
        assert_eq!(original.num_clusters_factor, deserialized.num_clusters_factor);
        assert_eq!(original.k, deserialized.k);
        assert_eq!(original.delta, deserialized.delta);
        assert_eq!(original.dataset_name, deserialized.dataset_name);
        assert!(matches!(deserialized.metrics_output, MetricsOutput::None));
    }
    
    #[test]
    fn test_clone() {
        let original = Config::new(
            512, 
            7.0, 
            25, 
            0.85, 
            "clone_test", 
            MetricsOutput::None
        );
        
        let cloned = original.clone();
        
        // Assert the cloned config matches the original
        assert_eq!(original.num_tables, cloned.num_tables);
        assert_eq!(original.num_clusters_factor, cloned.num_clusters_factor);
        assert_eq!(original.k, cloned.k);
        assert_eq!(original.delta, cloned.delta);
        assert_eq!(original.dataset_name, cloned.dataset_name);
        assert!(matches!(cloned.metrics_output, MetricsOutput::None));
    }
    
    #[test]
    fn test_different_metric_outputs() {
        // Test with different MetricsOutput variants
        
        // Create configs with different metric outputs
        let config1 = Config::new(1, 1.0, 10, 0.9, "test", MetricsOutput::DB);
        
        // Serialize and deserialize
        let serialized = serde_json::to_string(&config1).unwrap();
        let deserialized: Config = serde_json::from_str(&serialized).unwrap();
        
        // Verify metric output is preserved
        assert!(matches!(deserialized.metrics_output, MetricsOutput::DB));
    }
}