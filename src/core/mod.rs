pub mod config;
pub mod index;
pub mod errors;
pub mod gmm;
mod heap;

pub use config::Config;
pub use errors::{Result, ClusteredIndexError};