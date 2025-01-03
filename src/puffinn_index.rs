use core::slice;

use crate::puffinn_bindings::{CPUFFINN_index_create, CPUFFINN_index_insert_float, CPUFFINN_index_insert_uint32, CPUFFINN_index_rebuild, CPUFFINN_search_float, CPUFFINN_search_uint32, CPUFFINN};

use std::marker::PhantomData;
use std::ffi::CString;

/// Trait for data types supported by PUFFINN.
pub trait PuffinnDataType {
    /// Insert data into the index.
    unsafe fn insert_data(raw: *mut CPUFFINN, point: *const Self, dimension: i32);

    /// Search data in the index.
    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const Self,
        k: u32,
        recall: f32,
        dimension: i32,
    ) -> *mut u32;
}

/// Implementation for `f32` (Cosine Similarity).
impl PuffinnDataType for f32 {
    unsafe fn insert_data(raw: *mut CPUFFINN, point: *const Self, dimension: i32) {
        CPUFFINN_index_insert_float(raw, point as *mut f32, dimension);
    }

    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const Self,
        k: u32,
        recall: f32,
        dimension: i32,
    ) -> *mut u32 {
        CPUFFINN_search_float(raw, query as *mut f32, k, recall, dimension)
    }
}

/// Implementation for `u32` (Jaccard Similarity).
impl PuffinnDataType for u32 {
    unsafe fn insert_data(raw: *mut CPUFFINN, point: *const Self, dimension: i32) {
        CPUFFINN_index_insert_uint32(raw, point as *mut u32, dimension);
    }

    unsafe fn search_data(
        raw: *mut CPUFFINN,
        query: *const Self,
        k: u32,
        recall: f32,
        dimension: i32,
    ) -> *mut u32 {
        CPUFFINN_search_uint32(raw, query as *mut u32, k, recall, dimension)
    }
}

/// PUFFINN Index Structure
pub struct PuffinnIndex<T: PuffinnDataType> {
    raw: *mut CPUFFINN,
    _marker: PhantomData<T>,
}

/// Enum to represent similarity measures.
pub enum SimilarityMeasure {
    CosineSimilarity,
    JaccardSimilarity,
}

impl<T: PuffinnDataType> PuffinnIndex<T> {
    /// Create a new PUFFINN index
    /// 
    /// `dataset_args` depends on the similarity used. When [SimilarityMeasure::CosineSimilarity] is used, it specifies the dimension that all vectors must have. 
    /// When using [SimilarityMeasure::JaccardSimilarity] it specifies the universe size.
    /// 
    /// `memory_limit` is the number of bytes the index can use
    /// 
    /// # Examples
    ///
    /// ```
    /// let index = PuffinnIndex::<f32>::new(100, 1024, SimilarityMeasure::CosineSimilarity)
    /// ```
    pub fn new(
        dataset_args: i32,
        memory_limit: u64,
        similarity_measure: SimilarityMeasure,
    ) -> Result<Self, String> {
        // just for the C API to double check the type
        let dataset_type = match similarity_measure {
            SimilarityMeasure::CosineSimilarity => "cosine",
            SimilarityMeasure::JaccardSimilarity => "angular"
        };
        let dataset_type_cstr = CString::new(dataset_type)
            .map_err(|_| format!("Failed to convert dataset type '{}' to CString", dataset_type))?;
        
        // Call the C wrapper
        let raw = unsafe { CPUFFINN_index_create(dataset_type_cstr.as_ptr(), dataset_args, memory_limit) };
        
        if raw.is_null() {
            Err("Failed to create PUFFINN index".to_string())
        } else {
            Ok(Self {
                raw,
                _marker: PhantomData,
            })
        }
    }


    /// Insert a single data point into the index.
    pub fn insert(&mut self, point: &[T]) -> Result<(), String> {
        let dimension = point.len() as i32;
        unsafe {
            T::insert_data(self.raw, point.as_ptr(), dimension);
        }
        Ok(())
    }

    /// Rebuild the index using the currently inserted points.
    pub fn rebuild(&self) -> Result<(), String> {
        unsafe {
            CPUFFINN_index_rebuild(self.raw);
        }
        Ok(())
    }

    /// Search for the approximate k nearest neighbors to a query.
    /// 
    /// Returns the indices of the k nearest found neighbors. Indices are assigned incrementally to each point in the order they are inserted into the dataset, starting at 0. The result is ordered so that the most similar neighbor is first.
    pub fn search(&self, query: &[T], k: u32, recall: f32) -> Result<Vec<u32>, String> {
        let dimension = query.len() as i32;
        let result_ptr = unsafe {
            T::search_data(self.raw, query.as_ptr(), k, recall, dimension)
        };

        if result_ptr.is_null() {
            return Err("Search failed".to_string());
        }

        let results: Vec<u32> =
            unsafe { slice::from_raw_parts(result_ptr, k as usize).to_vec() };
        Ok(results)
    }
}
