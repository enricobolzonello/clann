use std::path::Path;

fn main() {
    // Define paths and flags
    let puffinn_include_dir = Path::new("puffinn/include");
    let c_api_dir = Path::new("c_api");
    let header_file = c_api_dir.join("c_binder.h");
    let cpp_file = c_api_dir.join("c_binder.cpp");

    // First, compile the C++ code using cc-rs
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(cpp_file)
        .include(puffinn_include_dir)
        .include(c_api_dir)
        .flag("-std=c++14")
        .flag("-march=native")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-O3")
        .flag("-fopenmp");

    // Print the actual commands being run (helpful for debugging)
    build.debug(true);
    
    // Attempt to compile
    println!("cargo:rerun-if-changed=c_api/c_binder.cpp");
    println!("cargo:rerun-if-changed=c_api/c_binder.h");
    build.compile("puffinn");

    // Now generate the Rust bindings
    let bindings = bindgen::Builder::default()
        .allowlist_function("^CPUFFINN_.*")
        .ctypes_prefix("cty")
        .use_core()
        .header(header_file.to_str().expect("Invalid header path"))
        .clang_arg(format!("-I{}", puffinn_include_dir.display()))
        .clang_arg(format!("-I{}", c_api_dir.display()))
        // Only pass C-compatible flags to bindgen
        .clang_arg("-Wall")
        .clang_arg("-Wextra")
        // Enable C++ processing
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        // Trust the C declarations as is
        .trust_clang_mangling(true)
        // Parse only the C-compatible declarations
        .generate_comments(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/puffinn_binds/puffinn_bindings.rs")
        .expect("Couldn't write bindings!");

    // Link against OpenMP
    println!("cargo:rustc-link-lib=gomp");
}