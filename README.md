<div align="center">

  <h1>CLANN</h1>

  <strong>CLANN: Clustered LSH-based Algorithm for the Nearest Neighbors problem</strong>
</div>

## Prerequisites
The algorithm is built on top of PUFFINN, which was built on C++. To achieve Rust integration, [bindgen](https://docs.rs/bindgen/latest/bindgen/) has been used, which, provided a C/C++ header file, receives Rust FFI code to call into C/C++ functions and use types defined in the header. But generating directly C++ binds is quite tricky since [not all features are supported](https://rust-lang.github.io/rust-bindgen/cpp.html), instead what has been done is a minimal C wrapper around the basic functions to then generate the bindings. To do this, some prerequisites are needed, namily:
- Clang 9.0 or greater
- A valid OpenMP installation, what worked for me is to install the development package, even though a OpenMP installation for GCC already exists: 
`sudo apt install libomp-dev `

At this point, all you need to do is do `cargo build` and it should automatically compile.