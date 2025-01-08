#[derive(Debug)]
pub enum ClusteredIndexError {
    ConfigError(String),
    DataError(String),
}
