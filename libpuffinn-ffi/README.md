# libpuffinn-ffi

C wrapper for the PUFFINN (Parameterless and Universal Fast FInding of Nearest Neighbors) library. This wrapper provides C bindings for the core functionality of PUFFINN, enabling its use in C applications and other languages through FFI.

## Features

### Similarity Measures
- [x] Cosine Similarity
- [ ] L2 (Euclidean) Similarity
- [x] Jaccard Similarity

### Index Operations
- [x] Index Creation
- [x] Data Insertion
- [x] Index Building
- [x] Searching
    - [x] k
    - [x] recall
    - [ ] filter_type
- [x] Serialization
    -[x] Base
    - [ ] Serialize chunks
- [ ] Get
- [ ] Change LSH function family
- [ ] Change Sketch function

## API Reference

### Index Management
```c
// Create a new index with specified dataset type ("angular" or "jaccard")
CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args);

// Load an index from HDF5 file
CPUFFINN* CPUFFINN_load_from_file(const char* file_name, const char* dataset_name);
```

### Data Operations
```c
// Insert a point into the index (for cosine similarity)
void CPUFFINN_index_insert_cosine(CPUFFINN* index, float* point, int dimension);

// Rebuild the index with specified number of hash tables
uint64_t CPUFFINN_index_rebuild(CPUFFINN* index, unsigned int num_maps);
```

### Search Operations
```c
// Search for nearest neighbors (for cosine similarity)
uint32_t* CPUFFINN_search_cosine(
    CPUFFINN* index,
    float* query,
    unsigned int k,
    float recall,
    float max_sim,
    int dimension
);
```

### Serialization
```c
// Save index to HDF5 file
void CPUFFINN_save_index(CPUFFINN* index, const char* file_name, int index_id);
```

### Metrics
```c
// Get and clear performance metrics
unsigned int CPUFFINN_get_distance_computations();
void CPUFFINN_clear_distance_computations();
```

## Usage Example

```c
#include "c_binder.h"

int main() {
    // Create a new cosine similarity index
    CPUFFINN* index = CPUFFINN_index_create("angular", 128);
    
    // Insert a vector
    float vector[128] = {...};
    CPUFFINN_index_insert_cosine(index, vector, 128);
    
    // Build the index
    uint64_t memory_used = CPUFFINN_index_rebuild(index, 10);
    
    // Search for nearest neighbors
    float query[128] = {...};
    uint32_t* results = CPUFFINN_search_cosine(index, query, 10, 0.9, 1.0, 128);
    
    // Save the index
    CPUFFINN_save_index(index, "index.h5", 0);
    
    // Free results when done
    free(results);
    
    return 0;
}
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Dependencies

- C++17 compiler
- OpenMP
- CMake (>= 3.10)
- HDF5

## License

Same as PUFFINN library (MIT License)
