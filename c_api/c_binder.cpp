#include "c_binder.h"

extern "C" {
    CPUFFINN* CPUFFINN_load_from_file(const char* file_name, const char* dataset_name) {
        // Open HDF5 file
        hid_t file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Failed to open HDF5 file");
        }

        // Open dataset
        hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
        if (dataset_id < 0) {
            H5Fclose(file_id);
            throw std::runtime_error(std::string("Failed to open dataset: ") + dataset_name);
        }

        // Read binary data into memory
        hid_t dataspace_id = H5Dget_space(dataset_id);
        hsize_t size;
        H5Sget_simple_extent_dims(dataspace_id, &size, nullptr);
        std::vector<uint8_t> buffer(size);
        H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

        // Close HDF5 handles
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);

        // Convert buffer to istream
        std::string buffer_str(buffer.begin(), buffer.end());
        std::istringstream* input_stream = new std::istringstream(buffer_str);

        // Create the PUFFINN index
        return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::CosineSimilarity>(*input_stream));
    }

    // Create a new index
    CPUFFINN* CPUFFINN_index_create(const char* dataset_type, int dataset_args) {

        if (strcmp("angular", dataset_type) == 0) {
            return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::CosineSimilarity>(dataset_args));
        }else if (strcmp("jaccard", dataset_type) == 0){
            return reinterpret_cast<CPUFFINN*>(new puffinn::Index<puffinn::JaccardSimilarity>(dataset_args));
        }else{
            std::cerr << "Error: Unsupported dataset type '" << dataset_type << "'. Only 'angular' is supported." << std::endl;
            return nullptr;
        }
      
    }

    // Rebuild the index
    uint64_t CPUFFINN_index_rebuild(CPUFFINN* index, unsigned int num_maps) {
        try{
            auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
            return cpp_index->rebuild(num_maps);
        } catch (...) {
            return 0;
        }
    }

    // Insert a point into the index
    void CPUFFINN_index_insert_cosine(CPUFFINN* index, float* point, int dimension) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        cpp_index->insert(std::vector<float>(point, point + dimension));
    }

    // Search in the index
    uint32_t* CPUFFINN_search_cosine(CPUFFINN* index, float* query, unsigned int k, float recall, float max_sim, int dimension) {
        if (!query || dimension <= 0) {
            std::cerr << "Error: Query is null or empty.\n";
            return nullptr;
        }
    
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        auto result = cpp_index->search(std::vector<float>(query, query + dimension), k, recall, max_sim);
    
        if (result.empty()) {
            uint32_t* empty_result = static_cast<uint32_t*>(malloc(sizeof(uint32_t)));
            if (!empty_result) {
                std::cerr << "Memory allocation failed for empty result!\n";
                return nullptr;
            }
            *empty_result = 0xFFFFFFFF;  // Sentinel value for empty result
            return empty_result;
        }
    
        uint32_t* c_result = static_cast<uint32_t*>(malloc(result.size() * sizeof(uint32_t)));
        if (!c_result) {
            std::cerr << "Memory allocation failed!\n";
            return nullptr;
        }
    
        std::memcpy(c_result, result.data(), result.size() * sizeof(uint32_t));
        return c_result;
    }    

    unsigned int CPUFFINN_get_distance_computations() {
        return puffinn::g_performance_metrics.get_distance_computations();
    }
    
    void CPUFFINN_clear_distance_computations() {
        puffinn::g_performance_metrics.clear();
    }

    void CPUFFINN_save_index(CPUFFINN* index, const char* file_name, int index_id) {
        auto cpp_index = reinterpret_cast<puffinn::Index<puffinn::CosineSimilarity>*>(index);
        
        // Open the existing HDF5 file in read-write mode
        hid_t file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0) {
            std::cerr << "Error opening HDF5 file: " << file_name << std::endl;
            return;
        }

        // Serialize the index into a string buffer
        std::stringstream buffer;
        cpp_index->serialize(buffer, false);
        std::string data = buffer.str();
        hsize_t data_size = data.size();

        std::string dataset_name = "index_" + std::to_string(index_id);

        if (H5Lexists(file_id, dataset_name.c_str(), H5P_DEFAULT) > 0) {
            // Delete the old dataset if it exists
            H5Ldelete(file_id, dataset_name.c_str(), H5P_DEFAULT);
        }

        // Create a new dataset for the index
        hid_t dataspace_id = H5Screate_simple(1, &data_size, nullptr);
        hid_t dataset_id = H5Dcreate(file_id, dataset_name.c_str(), H5T_NATIVE_UINT8, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (dataset_id < 0) {
            std::cerr << "Error creating dataset: " << dataset_name << std::endl;
            H5Sclose(dataspace_id);
            H5Fclose(file_id);
            return;
        }

        // Write the serialized data to the dataset
        H5Dwrite(dataset_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.c_str());

        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
    }
}
