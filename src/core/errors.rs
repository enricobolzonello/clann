use thiserror::Error;

pub type Result<T> = std::result::Result<T, ClusteredIndexError>;

#[derive(Debug, Error, PartialEq)]
pub enum ClusteredIndexError {
    #[error("Configuration Error: {0}")]
    ConfigError(String),

    #[error("Data Error: {0}")]
    DataError(String),

    #[error("Result DB Error: {0}")]
    ResultDBError(String),

    #[error("Invalid Assignment: {0} not found")]
    InvalidAssignment(usize),

    #[error("PUFFINN Creation Error: {0}")]
    PuffinnCreationError(String),

    #[error("PUFFINN Search Error: {0}")]
    PuffinnSearchError(String),

    #[error("Index Not Found Error")]
    IndexNotFound(),

    #[error("Index Out of Bounds: {0} out of {1} length")]
    IndexOutOfBounds(usize, usize),

    #[error("Index Mapping Error: {0}")]
    IndexMappingError(u32),

    #[error("Serialize Error: {0}")]
    SerializeError(String),
}
