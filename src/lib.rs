#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]

//! Clustered LSH-based Algorithm for the Nearest Neighbors problem
//! 
//! CLANN is an algorithm solving the Nearest Neighbors problem, built on top of PUFFINN. Rather than constructing a single index, the algorithm begins by dividing the dataset into clusters based on a user-defined number. It then builds a separate PUFFINN index for each cluster.
//! This approach, even though requires more memory and index building time, effectively cuts the hit distribution for the LSH function, ensuring that points that are far apart cannot collide. In classic LSH scenarios, it has been observed long tails of hits, due to the probabilistic nature of the function. Even though far points have low probability of colliding it was still not null, and the problem accentuated with queries far away from the dataset, where it approximates to a brute-force approach.
//! 

use core::{config, index::ClusteredIndex, ClusteredIndexError, Config, Result};

use metricdata::MetricData;
use puffinn_binds::IndexableSimilarity;

pub mod metricdata;
pub mod core;
pub mod puffinn_binds;

pub fn init<T: MetricData + IndexableSimilarity<T>>(data: T) -> Result<ClusteredIndex<T>> {
    init_with_config(data, Config::default())
}


pub fn init_with_config<T: MetricData + IndexableSimilarity<T>>(data: T, config: Config) -> Result<ClusteredIndex<T>> {
    ClusteredIndex::new(config, data)
}