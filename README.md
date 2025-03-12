<div align="center">

  <h1>CLANN</h1>

  <strong>CLANN: Clustered LSH-based Algorithm for the Nearest Neighbors problem</strong>

</div>

## Overview

CLANN is an algorithm for solving the Nearest Neighbors problem, built on top of PUFFINN (Parameterless and Universal FInding of Nearest Neighbors). Rather than constructing a single index, CLANN first divides the dataset into clusters and then builds a separate PUFFINN index for each cluster.

## Features

- **Similarity Measures**
  - Cosine Similarity

- **Search Options**
  - k-nearest neighbor search
  - Configurable recall targets

- **Performance Metrics**
  - Distance computation tracking
  - Memory usage monitoring
  - Build and search time measurements
  - Per-cluster statistics

- **Serialization Support**
  - HDF5-based storage
  - Versioned index format

## Prerequisites

The algorithm requires several dependencies for compilation and execution:

### Core Requirements
- Clang 9.0 or greater
- OpenMP installation
- HDF5 library
- CMake (>= 3.10)
- Rust toolchain (2021 edition or newer)

### Using NixOS (Alternative)
If you have Nix installed:
```bash
nix develop
```

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/clann.git
   cd clann
   ```

2. **Build the Project**
   ```bash
   cargo build --release
   ```

3. **Run Benchmark**, you can run comparisons between PUFFINN and CLANN in terms of distance computations, modify the parameters and the dataset in `benches/configs.json` and run:
   ```bash
   cargo bench --bench=distance_benches
   ```

## Usage

### Basic Example

```rust
use clann::{init_with_config, Config, MetricsOutput};
use ndarray::Array2;

fn main() {
    // Create configuration
    let config = Config{
        num_tables: 84,
        num_clusters_factor: 0.4,
        k: 10,
        delta: 0.9,
        dataset_name: "glove-25-angular".to_owned(),
        metrics_output: MetricsOutput::DB,
    };

    // Initialize and build index
    let data = Array2::random((10000, 128));
    let mut index = init_with_config(data, config).unwrap();
    build(&mut index).unwrap();

    // Perform search
    let query = vec![0.1; 128];
    let results = search(&mut index, &query).unwrap();
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style
- Testing requirements
- Pull request process
- Development setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.