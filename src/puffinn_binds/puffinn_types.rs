use ndarray::Data;

use crate::metricdata::{AngularData, EuclideanData, MetricData};

use super::puffinn_bindings::{CPUFFINN_index_insert_float, CPUFFINN_index_insert_uint32, CPUFFINN_search_float, CPUFFINN_search_uint32, CPUFFINN};

/// This trait extends [`MetricData`] enabling the insertion of the data into the PUFFINN index.
pub trait IndexableSimilarity<M: MetricData> {

    /// Returns the similarity type as understood by PUFFINN (e.g., "cosine", "angular").
    fn similarity_type(&self) -> &'static str;

    /// Inserts a data point into the PUFFINN index.
    unsafe fn insert_data(
        raw: *mut CPUFFINN,
        point: *const M::DataType,
        dimension: i32,
    );

    /// Searches for the nearest neighbors using the PUFFINN index.
    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const M::DataType,
        k: u32,
        recall: f32,
        dimension: i32,
    ) -> *mut u32;
}

impl<S: Data<Elem = f32>, M: MetricData> IndexableSimilarity<M> for AngularData<S> {

    fn similarity_type(&self) -> &'static str {
        "cosine"
    }

    unsafe fn insert_data(
        raw: *mut CPUFFINN,
        point: *const M::DataType,
        dimension: i32,
    ) {
        CPUFFINN_index_insert_float(raw, point as *mut f32, dimension);
    }

    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const M::DataType,
        k: u32,
        recall: f32,
        dimension: i32,
    ) -> *mut u32 {
        CPUFFINN_search_float(raw, query as *mut f32, k, recall, dimension)
    }
}
