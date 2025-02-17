/* automatically generated by rust-bindgen 0.71.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CPUFFINN {
    _unused: [u8; 0],
}
unsafe extern "C" {
    pub fn CPUFFINN_load_from_file(
        file_name: *const cty::c_char,
        dataset_name: *const cty::c_char,
    ) -> *mut CPUFFINN;
}
unsafe extern "C" {
    pub fn CPUFFINN_index_create(
        dataset_type: *const cty::c_char,
        dataset_args: cty::c_int,
    ) -> *mut CPUFFINN;
}
unsafe extern "C" {
    pub fn CPUFFINN_index_rebuild(index: *mut CPUFFINN, num_maps: cty::c_uint) -> u64;
}
unsafe extern "C" {
    pub fn CPUFFINN_index_insert_cosine(
        index: *mut CPUFFINN,
        point: *mut f32,
        dimension: cty::c_int,
    );
}
unsafe extern "C" {
    pub fn CPUFFINN_search_cosine(
        index: *mut CPUFFINN,
        query: *mut f32,
        k: cty::c_uint,
        recall: f32,
        max_sim: f32,
        dimension: cty::c_int,
    ) -> *mut u32;
}
unsafe extern "C" {
    pub fn CPUFFINN_get_distance_computations() -> cty::c_uint;
}
unsafe extern "C" {
    pub fn CPUFFINN_clear_distance_computations();
}
unsafe extern "C" {
    pub fn CPUFFINN_save_index(
        index: *mut CPUFFINN,
        file_name: *const cty::c_char,
        index_number: cty::c_int,
    );
}
