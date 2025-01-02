#include "c_binder.h"

extern "C" {
    // Create a new index
    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args, uint64_t memory_limit) {
        // TODO: handle dataset type (for now only angular is supported)
        if (strcmp("angular", dataset_type) != 0) {
            std::cerr << "Error: Unsupported dataset type '" << dataset_type << "'. Only 'angular' is supported." << std::endl;
            return nullptr;
        }

        return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::CosineSimilarity>(dataset_args, memory_limit));
    }

    // Insert a point into the index
    void CPUFFINN_index_insert(CPUFFINN* index, float* point, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        cpp_index->insert(std::vector<float>(point, point + dimension));
    }

    // Rebuild the index
    void CPUFFINN_index_rebuild(CPUFFINN* index) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        cpp_index->rebuild();
    }

    // Search in the index
    uint32_t* CPUFFINN_search(CPUFFINN* index, float* query, unsigned int k, float recall, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        auto result = cpp_index->search(std::vector<float>(query, query + dimension), k, recall);

        uint32_t* c_result = (uint32_t*)malloc(result.size() * sizeof(uint32_t));
        std::copy(result.begin(), result.end(), c_result);
        return c_result;
    }
}
