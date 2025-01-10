use super::puffinn_bindings::{
    CPUFFINN_index_create, CPUFFINN_index_rebuild, CPUFFINN,
};
use super::puffinn_types::IndexableSimilarity;
use crate::metricdata::MetricData;
use std::ffi::CString;

pub struct PuffinnIndex {
    raw: *mut CPUFFINN,
}

impl PuffinnIndex {
    pub fn new<M: MetricData + IndexableSimilarity<M>>(metric_data: &M, memory_limit: usize) -> Result<Self, String> {
        let dataset_type = metric_data.similarity_type();
        let dataset_type_cstr = CString::new(dataset_type)
            .map_err(|_| format!("Failed to convert dataset type '{}' to CString", dataset_type))?;

        let raw = unsafe {
            CPUFFINN_index_create(
                dataset_type_cstr.as_ptr(),
                metric_data.dimensions() as i32,
                memory_limit as u64,
            )
        };

        if raw.is_null() {
            return Err("Failed to create PUFFINN index".to_string());
        }

        let index = Self { raw };

        // Iterate over the data points and insert them.
        for i in 0..metric_data.num_points() {
            let point = metric_data.get_point(i).to_owned();
            unsafe {
                M::insert_data(index.raw, point.as_ptr(), metric_data.dimensions() as i32);
            }
        }

        // Rebuild the index after inserting the points.
        unsafe {
            CPUFFINN_index_rebuild(index.raw);
        }

        Ok(index)
    }

    pub fn search<M: MetricData + IndexableSimilarity<M>>(
        &self,
        query: &[M::DataType],
        k: usize,
        recall: f32,
    ) -> Result<Vec<u32>, String> {

        unsafe {
            
            let results_ptr = M::search_data(
                self.raw,
                query.as_ptr(),
                k as u32,
                recall,
                query.len() as i32,
            );

            if results_ptr.is_null() {
                return Err("Search failed: returned null pointer.".to_string());
            }

            let results_slice = std::slice::from_raw_parts(results_ptr, k);
            let results = results_slice.to_vec();

            Ok(results)
            
        }
    }
}

