use std::{path::Path, process::Command};

fn main() {
    // Get the current Git commit hash
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .expect("Failed to execute git command");

    if output.status.success() {
        let git_hash = String::from_utf8(output.stdout).unwrap_or_default();
        println!("cargo:rustc-env=GIT_COMMIT_HASH={}", git_hash.trim());
    } else {
        eprintln!("Failed to get Git commit hash");
        println!("cargo:rustc-env=GIT_COMMIT_HASH=unknown");
    }

    // Define paths and flags
    let hdf5 = pkg_config::Config::new()
        .atleast_version("1.10")
        .probe("hdf5")
        .expect("Failed to find HDF5");
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
        .flag("-fopenmp")
        .flag("-lhdf5_cpp");
    for path in &hdf5.include_paths {
        build.include(path);
    }

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
        .clang_arg("-Wall")
        .clang_arg("-Wextra")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .clang_args(
            &hdf5.include_paths.iter().map(|path| format!("-I{}", path.display())).collect::<Vec<_>>()
        )
        .trust_clang_mangling(true)
        .generate_comments(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/puffinn_binds/puffinn_bindings.rs")
        .expect("Couldn't write bindings!");

    // Link against OpenMP
    println!("cargo:rustc-link-lib=gomp");

    // rebuild if there is a commit
    println!("cargo:rerun-if-changed=.git/HEAD");
}