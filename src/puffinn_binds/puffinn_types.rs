use log::{error, warn};
use ndarray::Data;

use crate::metricdata::{AngularData, MetricData};

use super::puffinn_sys::{CPUFFINN_index_insert_cosine, CPUFFINN_search_cosine, CPUFFINN};

/// This trait extends [`MetricData`] enabling the insertion of the data into the PUFFINN index.
pub trait IndexableSimilarity<M: MetricData> {

    /// Returns the similarity type as understood by PUFFINN (e.g., "cosine", "angular").
    fn similarity_type(&self) -> &'static str;

    /// Inserts a data point into the PUFFINN index.
    /// 
    /// # Safety
    /// Uses a C++ library
    unsafe fn insert_data(
        raw: *mut CPUFFINN,
        point: *const M::DataType,
        dimension: i32,
    );

    /// Searches for the nearest neighbors using the PUFFINN index.
    /// 
    /// # Safety
    /// Uses a C++ library
    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const M::DataType,
        k: u32,
        recall: f32,
        max_sim: f32,
        dimension: i32,
    ) -> *mut u32;

    fn convert_to_sim(max_dist: f32) -> f32;
}

impl<S: Data<Elem = f32> + ndarray::RawDataClone, M: MetricData> IndexableSimilarity<M> for AngularData<S> {

    fn similarity_type(&self) -> &'static str {
        "angular"
    }

    unsafe fn insert_data(
        raw: *mut CPUFFINN,
        point: *const M::DataType,
        dimension: i32,
    ) {
        CPUFFINN_index_insert_cosine(raw, point as *mut f32, dimension);
    }

    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const M::DataType,
        k: u32,
        recall: f32,
        max_sim: f32,
        dimension: i32,
    ) -> *mut u32 {
        if query.is_null() || dimension <= 0 {
            warn!("Empty query or wrong dimensions");
            return std::ptr::null_mut();
        }
    
        let result_ptr = CPUFFINN_search_cosine(raw, query as *mut f32, k, recall, max_sim, dimension);
    
        if result_ptr.is_null() {
            error!("Search failed, received null pointer");
            return std::ptr::null_mut();
        }
    
        result_ptr
    }    

    fn convert_to_sim(distance: f32) -> f32 {
        1.0 - distance / 2.0
    }
}
