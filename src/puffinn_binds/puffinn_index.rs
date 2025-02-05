use super::puffinn_bindings::{
    CPUFFINN_clear_distance_computations, CPUFFINN_get_distance_computations,
    CPUFFINN_index_create, CPUFFINN_index_rebuild, CPUFFINN_load_from_file, CPUFFINN_save_index,
    CPUFFINN,
};
use super::puffinn_types::IndexableSimilarity;
use crate::metricdata::MetricData;
use std::ffi::CString;

pub struct PuffinnIndex {
    raw: *mut CPUFFINN,
}

impl PuffinnIndex {
    pub fn new<M: MetricData + IndexableSimilarity<M>>(
        metric_data: &M,
        memory_limit: usize,
    ) -> Result<Self, String> {
        let dataset_type = metric_data.similarity_type();
        let dataset_type_cstr = CString::new(dataset_type).map_err(|_| {
            format!(
                "Failed to convert dataset type '{}' to CString",
                dataset_type
            )
        })?;

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
            let r = CPUFFINN_index_rebuild(index.raw);
            if r == 1 {
                return Err("Failed to create PUFFINN index, insufficient memory".to_string());
            }
        }

        Ok(index)
    }

    pub fn new_from_file(file_path: &str, dataset_name: &str) -> Result<Self, String> {
        let file_path_cstr = CString::new(file_path)
            .map_err(|_| format!("Failed to convert dataset type '{}' to CString", file_path))?;
        let dataset_name_cstr = CString::new(dataset_name).map_err(|_| {
            format!(
                "Failed to convert dataset type '{}' to CString",
                dataset_name
            )
        })?;

        let raw =
            unsafe { CPUFFINN_load_from_file(file_path_cstr.as_ptr(), dataset_name_cstr.as_ptr()) };

        Ok(Self { raw })
    }

    pub fn search<M: MetricData + IndexableSimilarity<M>>(
        &self,
        query: &[M::DataType],
        k: usize,
        max_dist: f32,
        recall: f32,
    ) -> Result<Vec<u32>, String> {
        // convert distance into similarity
        let max_sim = 1.0 - max_dist / 2.0;

        unsafe {
            let results_ptr = M::search_data(
                self.raw,
                query.as_ptr(),
                k as u32,
                recall,
                max_sim,
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

    pub fn save_to_file(&self, file_path: &str, index_id: usize) -> Result<(), String> {
        let file_path_cstring = CString::new(file_path)
            .map_err(|_| format!("Failed to convert file name '{}' to CString", file_path))?;

        unsafe {
            CPUFFINN_save_index(self.raw, file_path_cstring.as_ptr(), index_id as i32);
        }

        Ok(())
    }
}

pub fn get_distance_computations() -> u32 {
    unsafe { CPUFFINN_get_distance_computations() }
}

pub fn clear_distance_computations() {
    unsafe {
        CPUFFINN_clear_distance_computations();
    }
}
