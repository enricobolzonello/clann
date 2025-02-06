#pragma once

#include <vector>

#include "puffinn/format/generic.hpp"

namespace puffinn {
    struct RealVectorFormat {
        using Type = float;
        using Args = unsigned int;
        // 256 bit vectors
        const static unsigned int ALIGNMENT = 256/8;

        static unsigned int storage_dimensions(Args dimensions) {
            return dimensions;
        }


        static uint64_t inner_memory_usage(Type&) {
            return 0;
        }

        static void store(
            const std::vector<float>& input,
            Type* storage,
            DatasetDescription<RealVectorFormat> dataset
        ) {
            if (input.size() != dataset.args) {
                throw std::invalid_argument("input.size()");
            }
            for (size_t i=0; i < dataset.args; i++) {
                storage[i] = input[i];
            }
            for (size_t i=dataset.args; i < dataset.storage_len; i++) {
                storage[i] = 0.0;
            }
        }

         static void serialize_args(std::ostream& out, const Args& args) {
            out.write(reinterpret_cast<const char*>(&args), sizeof(Args));
        }

        static void deserialize_args(std::istream& in, Args* args) {
            in.read(reinterpret_cast<char*>(args), sizeof(Args));
        }

        static void serialize_type(std::ostream& out, const Type& type) {
            out.write(reinterpret_cast<const char*>(&type), sizeof(Type));
        }

        static void deserialize_type(std::istream& in, Type* type) {
            in.read(reinterpret_cast<char*>(type), sizeof(Type));
        }

        static void free(Type&) {}



        static std::vector<float> generate_random(unsigned int dimensions) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            auto& generator = get_default_random_generator();
            std::vector<float> values;
            for (unsigned int i=0; i<dimensions; i++) {
                values.push_back(normal_distribution(generator));
            }
            return values;
        }
    

        static std::vector<float> generate_random_range(unsigned int dimensions, std::pair<float, float> range){
            std::normal_distribution<float> normal_distribution(range.first, range.second);
            auto& generator = get_default_random_generator();
            std::vector<float> values;
            for(unsigned int i=0; i<dimensions;i++){
                values.push_back(normal_distribution(generator));
            }
            return values;
        }
        

    };
    template <>
    std::vector<float> convert_stored_type<RealVectorFormat, std::vector<float>>(
        typename RealVectorFormat::Type* storage,
        DatasetDescription<RealVectorFormat> dataset
    ) {
        std::vector<float> res;
        res.reserve(dataset.args);
        for (size_t i=0; i < dataset.args; i++) {
            res.push_back(storage[i]);
        }
        return res;
    }
}
