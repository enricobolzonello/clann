use serde::{Deserialize, Serialize};

/// Parameters for the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum main memory total usage for the index in bytes
    pub memory_limit: usize,

    /// Number of clusters
    pub k: usize
}

impl Default for Config {
    fn default() -> Self {
        Self { 
            memory_limit: 1073741824,   // 1 Gb 
            k: 5 
        }
    }
}

impl Config {
    pub fn new(
        memory_limit: usize,
        k: usize
    ) -> Self {
        Self{
            memory_limit,
            k
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.k < 2 {
            return Err("Clusters must be at least 2".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        
        // Check default values
        assert_eq!(config.memory_limit, 1073741824); // 1 GB
        assert_eq!(config.k, 5);
    }

    #[test]
    fn test_new_config() {
        let config = Config::new(2048, 10);
        
        // Check custom values
        assert_eq!(config.memory_limit, 2048);
        assert_eq!(config.k, 10);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = Config::new(2048, 10);
        
        // Validate should succeed for valid config
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_config() {
        let config = Config::new(2048, 1);
        
        // Validate should fail when k < 2
        let result = config.validate();
        assert_eq!(result, Err("Clusters must be at least 2".to_string()));
    }

    #[test]
    fn test_serialize_config() {
        let config = Config::new(2048, 10);
        
        // Check if it can serialize and deserialize
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&serialized).unwrap();
        
        // Assert the deserialized config matches the original
        assert_eq!(config.memory_limit, deserialized.memory_limit);
        assert_eq!(config.k, deserialized.k);
    }
}