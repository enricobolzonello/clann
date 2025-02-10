use ndarray::Data;

use crate::metricdata::{AngularData, EuclideanData, MetricData};

use super::puffinn_bindings::{CPUFFINN_index_insert_cosine, CPUFFINN_index_insert_l2, CPUFFINN_search_cosine, CPUFFINN_search_l2, CPUFFINN};

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
        max_sim: f32,
        dimension: i32,
    ) -> *mut u32;

    fn convert_to_sim(distance: f32) -> f32;
}

impl<S: Data<Elem = f32>, M: MetricData> IndexableSimilarity<M> for AngularData<S> {

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
        CPUFFINN_search_cosine(raw, query as *mut f32, k, recall, max_sim, dimension)
    }

    fn convert_to_sim(distance: f32) -> f32 {
        1.0 - distance / 2.0
    }
}

impl<S: Data<Elem = f32>, M: MetricData> IndexableSimilarity<M> for EuclideanData<S> {
    fn similarity_type(&self) -> &'static str {
        "euclidean"
    }

    unsafe fn insert_data(
        raw: *mut CPUFFINN,
        point: *const <M as MetricData>::DataType,
        dimension: i32,
    ) {
        CPUFFINN_index_insert_l2(raw, point as *mut f32, dimension);
    }

    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const <M as MetricData>::DataType,
        k: u32,
        recall: f32,
        max_sim: f32,
        dimension: i32,
    ) -> *mut u32 {
        CPUFFINN_search_l2(raw, query as *mut f32, k, recall, max_sim, dimension)
    }

    fn convert_to_sim(distance: f32) -> f32 {
        1.0 / (distance + 1.0)
    }
}
