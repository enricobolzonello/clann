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
}
