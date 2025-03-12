use super::puffinn_sys::{
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
        num_maps: usize,
    ) -> Result<(Self, usize), String> {
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
                metric_data.dimensions() as i32
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
        let memory;
        unsafe {
            let r = CPUFFINN_index_rebuild(index.raw, num_maps as u32);
            if r == 0 {
                return Err("Failed to create PUFFINN index, insufficient memory".to_string());
            }
            memory = r;
        }

        Ok((index, memory as usize))
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
        let max_sim = M::convert_to_sim(max_dist);

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

            let first_value = *results_ptr;

            if first_value == 0xFFFFFFFF {
                libc::free(results_ptr as *mut libc::c_void);
                return Ok(Vec::new());
            }

            let mut results = Vec::new();
            let mut offset = 0;

            while offset < k {
                let val = *(results_ptr.add(offset));
                results.push(val);
                offset += 1;
            }

            libc::free(results_ptr as *mut libc::c_void);
            Ok(results)
        }
    }

    pub(crate) fn save_to_file(&self, file_path: &str, index_id: usize) -> Result<(), String> {
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

pub(crate) fn clear_distance_computations() {
    unsafe {
        CPUFFINN_clear_distance_computations();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metricdata::AngularData;
    use crate::utils::{brute_force_search, generate_random_unit_vectors, load_hdf5_dataset};

    #[test]
    fn test_angular_create_index() {
        let hdf5_dataset = load_hdf5_dataset("./datasets/glove-25-angular.hdf5").unwrap();
        let data = AngularData::new(hdf5_dataset.dataset_array);
        let num_maps = 84;

        let index = PuffinnIndex::new(&data, num_maps);
        assert!(index.is_ok(), "Failed to create PuffinnIndex");
    }

    #[test]
    fn test_angular_search_index() {
        let hdf5_dataset = load_hdf5_dataset("./datasets/glove-25-angular.hdf5").unwrap();
        let data: AngularData<ndarray::OwnedRepr<f32>> = AngularData::new(hdf5_dataset.dataset_array);
        let num_maps = 84;
        let (index, _memory) = PuffinnIndex::new(&data, num_maps).unwrap();

        let binding = hdf5_dataset.dataset_queries.row(0);
        let query = binding.as_slice().unwrap();
        let k = 10;
        let max_dist = 1.0;
        let recall = 0.9;

        let results =
            index.search::<AngularData<ndarray::OwnedRepr<f32>>>(query, k, max_dist, recall);
        assert!(results.is_ok(), "Search failed");
        assert_eq!(results.unwrap().len(), k, "Search did not return k results");
    }

    #[test]
    fn test_puffinn_angular_search() {
        let n = 1000;
        let dimensions = 25;
        let data_raw = generate_random_unit_vectors(n, dimensions);
        let data = AngularData::new(data_raw.clone());
        let num_maps = 40;

        let (index, _memory) = PuffinnIndex::new(&data, num_maps).expect("Failed to create PuffinnIndex");

        let num_samples = 100;
        let recalls = [0.2, 0.5, 0.95];
        let ks = [1, 10];

        for &k in &ks {
            for &recall in &recalls {
                let mut num_correct = 0;
                let adjusted_k = k.min(n);
                let expected_correct = (recall * adjusted_k as f32 * num_samples as f32) as usize;

                for _ in 0..num_samples {
                    let query_raw = generate_random_unit_vectors(1, dimensions);
                    let binding = query_raw.row(0);
                    let query = binding.as_slice().unwrap();

                    let exact = brute_force_search(&data, query, k);
                    let approx = index
                        .search::<AngularData<ndarray::OwnedRepr<f32>>>(query, k, 1.0, recall)
                        .expect("Search failed");

                    assert_eq!(
                        approx.len(),
                        adjusted_k,
                        "Approximate search returned incorrect number of results"
                    );

                    num_correct += exact.iter().filter(|&&i| approx.contains(&i)).count();
                }

                assert!(
                    num_correct >= (0.8 * expected_correct as f32) as usize,
                    "Recall {} too low for k = {}",
                    recall,
                    k
                );
            }
        }
    }
}
