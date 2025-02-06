#pragma once

#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/tensor.hpp"

namespace puffinn {
    template <typename T, typename hashType>
    static std::unique_ptr<HashSourceArgs<T, hashType>> deserialize_hash_args(std::istream& in) {
        HashSourceType type;
        in.read(reinterpret_cast<char*>(&type), sizeof(HashSourceType));
        switch (type) {
            case HashSourceType::Independent:
                return std::make_unique<IndependentHashArgs<T, hashType>>(in);
            case HashSourceType::Pool:
                return std::make_unique<HashPoolArgs<T, hashType>>(in);
            case HashSourceType::Tensor:
                return std::make_unique<TensoredHashArgs<T, hashType>>(in);
            default:
                throw std::invalid_argument("hash source type");
        }
    }
}
