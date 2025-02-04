#include "../puffinn/include/puffinn.hpp"
#include <string.h>
#include <iostream>

extern "C" {
    struct CPUFFINN;
    typedef struct CPUFFINN CPUFFINN;

    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args, uint64_t memory_limit);
    int CPUFFINN_index_rebuild(CPUFFINN* index);

    // For float data (angular)
    void CPUFFINN_index_insert_float(CPUFFINN* index, float* point, int dimension);
    uint32_t* CPUFFINN_search_float(CPUFFINN* index, float* query, unsigned int k, float recall, float max_sim, int dimension);

    // For uint32_t data (jaccard)
    void CPUFFINN_index_insert_uint32(CPUFFINN* index, uint32_t* point, int dimension);
    uint32_t* CPUFFINN_search_uint32(CPUFFINN* index, uint32_t* query, unsigned int k, float recall, float max_sim, int dimension);

    unsigned int CPUFFINN_get_distance_computations();
    void CPUFFINN_clear_distance_computations();
}