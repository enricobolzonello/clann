#include "../puffinn/include/puffinn.hpp"
#include <string.h>
#include <iostream>

extern "C" {
    struct CPUFFINN;
    typedef struct CPUFFINN CPUFFINN;

    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args, uint64_t memory_limit);
    void CPUFFINN_index_insert(CPUFFINN* index, float* point, int dimension);
    void CPUFFINN_index_rebuild(CPUFFINN* index);
    uint32_t* CPUFFINN_search(CPUFFINN* index, float* query, unsigned int k, float recall, int dimension);
}