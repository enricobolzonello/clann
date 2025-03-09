pub(crate) mod config;
pub(crate) mod index;
pub(crate) mod errors;
pub(crate) mod gmm;
mod heap;

pub use config::{Config, MetricsOutput, MetricsGranularity};
pub use errors::{Result, ClusteredIndexError};