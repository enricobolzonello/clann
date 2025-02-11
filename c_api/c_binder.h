#include "../puffinn/include/puffinn.hpp"
#include <string.h>
#include <iostream>
#include <hdf5.h>
#include <vector>
#include <sstream>

#define EMPTY_RESULT_SENTINEL 0xFFFFFFFF

extern "C" {
    struct CPUFFINN;
    typedef struct CPUFFINN CPUFFINN;

    CPUFFINN* CPUFFINN_load_from_file(const char* file_name, const char* dataset_name);

    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args, uint64_t memory_limit);
    int CPUFFINN_index_rebuild(CPUFFINN* index);

    // For float data (angular)
    void CPUFFINN_index_insert_cosine(CPUFFINN* index, float* point, int dimension);
    uint32_t* CPUFFINN_search_cosine(CPUFFINN* index, float* query, unsigned int k, float recall, float max_sim, int dimension);

    unsigned int CPUFFINN_get_distance_computations();
    void CPUFFINN_clear_distance_computations();

    void CPUFFINN_save_index(CPUFFINN* index, const char* file_name, int index_number);
}