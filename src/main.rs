#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CString;

use cty::c_char;

mod puffinn_bindings;

fn main() {
    let angular_str = CString::new("angular").unwrap();
    let angular_c: *const c_char = angular_str.as_ptr() as *const c_char;

    unsafe {
        let _temp = puffinn_bindings::CPUFFINN_index_create(angular_c, 25, 1);
    }

    println!("done");
}
