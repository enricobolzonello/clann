/* automatically generated by rust-bindgen 0.71.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CPUFFINN {
    _unused: [u8; 0],
}
unsafe extern "C" {
    pub fn CPUFFINN_index_create(
        dataset_type: *const cty::c_char,
        dataset_args: cty::c_int,
        memory_limit: u64,
    ) -> *mut CPUFFINN;
}
unsafe extern "C" {
    pub fn CPUFFINN_index_rebuild(index: *mut CPUFFINN);
}
unsafe extern "C" {
    pub fn CPUFFINN_index_insert_float(
        index: *mut CPUFFINN,
        point: *mut f32,
        dimension: cty::c_int,
    );
}
unsafe extern "C" {
    pub fn CPUFFINN_search_float(
        index: *mut CPUFFINN,
        query: *mut f32,
        k: cty::c_uint,
        recall: f32,
        dimension: cty::c_int,
    ) -> *mut u32;
}
unsafe extern "C" {
    pub fn CPUFFINN_index_insert_uint32(
        index: *mut CPUFFINN,
        point: *mut u32,
        dimension: cty::c_int,
    );
}
unsafe extern "C" {
    pub fn CPUFFINN_search_uint32(
        index: *mut CPUFFINN,
        query: *mut u32,
        k: cty::c_uint,
        recall: f32,
        dimension: cty::c_int,
    ) -> *mut u32;
}
