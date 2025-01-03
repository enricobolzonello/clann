#include "c_binder.h"

extern "C" {
    // Create a new index
    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args, uint64_t memory_limit) {

        if (strcmp("angular", dataset_type) == 0) {
            return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::CosineSimilarity>(dataset_args, memory_limit));
        }else if (strcmp("jaccard", dataset_type) == 0){
            return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::JaccardSimilarity>(dataset_args, memory_limit));
        }else{
            std::cerr << "Error: Unsupported dataset type '" << dataset_type << "'. Only 'angular' is supported." << std::endl;
            return nullptr;
        }
      
    }

    // Rebuild the index
    void CPUFFINN_index_rebuild(CPUFFINN* index) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        cpp_index->rebuild();
    }

    // Insert a point into the index
    void CPUFFINN_index_insert_float(CPUFFINN* index, float* point, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        cpp_index->insert(std::vector<float>(point, point + dimension));
    }

    // Search in the index
    uint32_t* CPUFFINN_search_float(CPUFFINN* index, float* query, unsigned int k, float recall, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        auto result = cpp_index->search(std::vector<float>(query, query + dimension), k, recall);

        uint32_t* c_result = (uint32_t*)malloc(result.size() * sizeof(uint32_t));
        std::copy(result.begin(), result.end(), c_result);
        return c_result;
    }

    // Insert a point into the index
    void CPUFFINN_index_insert_uint32(CPUFFINN* index, uint32_t* point, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::JaccardSimilarity>*>(index);
        cpp_index->insert(std::vector<uint32_t>(point, point + dimension));
    }

    // Search in the index
    uint32_t* CPUFFINN_search_uint32(CPUFFINN* index, uint32_t* query, unsigned int k, float recall, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::JaccardSimilarity>*>(index);
        auto result = cpp_index->search(std::vector<uint32_t>(query, query + dimension), k, recall);

        uint32_t* c_result = (uint32_t*)malloc(result.size() * sizeof(uint32_t));
        std::copy(result.begin(), result.end(), c_result);
        return c_result;
    }
}
