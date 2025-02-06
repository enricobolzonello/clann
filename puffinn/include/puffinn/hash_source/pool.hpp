#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"

namespace puffinn {
    // A pool of hash functions that can be shared.
    // These functions can be mixed to produce different hashes, which means that fewer hash
    // computations are needed. However if the pool contains too few hash functions, it will
    // perform worse.
    template <typename T, typename hashType>
    class HashPool : public HashSource<T, hashType> {
        T hash_family;
        std::vector<typename T::Function> hash_functions;
        std::vector<std::vector<unsigned int>> indices;
        unsigned int num_tables;
        uint_fast8_t bits_per_function;
        unsigned int bits_per_hasher;
        unsigned int current_sampling_rep = 0;
        unsigned int bits_to_cut;

    public:
        HashPool(
            DatasetDescription<typename T::Sim::Format> desc,
            typename T::Args args,
            unsigned int num_functions,
            unsigned int num_tables,
            unsigned int bits_per_hasher
        )
          : hash_family(desc, args),
            num_tables(num_tables),
            bits_per_function(hash_family.bits_per_function()),
            bits_per_hasher(bits_per_hasher)
        {
            num_functions /= hash_family.bits_per_function();
            hash_functions.reserve(num_functions);
            for (unsigned int i=0; i < num_functions; i++) {
                hash_functions.push_back(hash_family.sample());
            }

            auto& rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, num_functions-1);
            
            indices.reserve(num_tables);
            for (size_t rep = 0; rep < num_tables; rep++) {
                std::vector<unsigned int> rep_indices;
                rep_indices.reserve(bits_per_hasher);
                for (size_t i=0; i < bits_per_hasher; i += bits_per_function) {
                    rep_indices.push_back(random_idx(rand_gen));
                }
                indices.push_back(rep_indices);
            }

            bits_to_cut = bits_per_function * indices[0].size() - bits_per_hasher;
        }

        HashPool(std::istream& in)
          : hash_family(in)
        {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            hash_functions.reserve(len);
            for (size_t i=0; i < len; i++) {
                hash_functions.emplace_back(in);
            }
            size_t len_indices;
            in.read(reinterpret_cast<char*>(&len_indices), sizeof(size_t));
            for (size_t i=0; i < len_indices; i++) {
                std::vector<unsigned int> rep_indices;
                size_t len_index_array;
                in.read(reinterpret_cast<char*>(&len_index_array), sizeof(size_t));
                for (size_t j=0; j < len_index_array; j++) {
                    unsigned int idx;
                    in.read(reinterpret_cast<char*>(&idx), sizeof(unsigned int));
                    rep_indices.push_back(idx);
                }
                indices.push_back(rep_indices);
            }

            in.read(reinterpret_cast<char*>(&num_tables), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits_per_function), sizeof(uint_fast8_t));
            in.read(reinterpret_cast<char*>(&bits_per_hasher), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&current_sampling_rep), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits_to_cut), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            hash_family.serialize(out);
            size_t len = hash_functions.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            for (auto& h : hash_functions) {
                h.serialize(out);
            }
            size_t len_indices = indices.size();
            out.write(reinterpret_cast<char*>(&len_indices), sizeof(size_t));
            for (auto& index_vec : indices) {
                size_t len_index_vec = index_vec.size();
                out.write(reinterpret_cast<char*>(&len_index_vec), sizeof(size_t));
                for (auto i : index_vec) {
                    out.write(reinterpret_cast<const char*>(&i), sizeof(unsigned int));
                }
            }
            out.write(reinterpret_cast<const char*>(&num_tables), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits_per_function), sizeof(uint_fast8_t));
            out.write(reinterpret_cast<const char*>(&bits_per_hasher), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&current_sampling_rep), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits_to_cut), sizeof(unsigned int));
        }


        //?This was apparently never used anywhere?
        // hashType concatenate_hash(
        //     const std::vector<unsigned int>& indices,
        //     const uint64_t* hashes //hashes are always 64bit unsigned ints
        // ) const {
        //     hashType res;
        //     res.concatenate_hashes(indices, hashes, bits_per_function);  
        //     return res;
        // }

        unsigned int get_size() const {
            return hash_functions.size();
        }

        uint_fast8_t get_bits_per_function() const {
            return bits_per_function;
        }

        uint_fast8_t get_bits_per_hasher() const {
            return bits_per_hasher;
        }

        void hash_repetitions(
            const typename T::Sim::Format::Type * const input,
            std::vector<hashType> & output
        ) const {
            output.clear();

            // TODO: remove this allocation and reuse the scratch space
            std::vector<uint64_t> pool; //FIX ME, I'm not sure if this should be another type, it hinges on whether we want hashType to just be a concatenation of hashes
                                        //Where do we put the abstraction, should hashType be able to correctly store the hash without knowledge of the hashfunction
                                        //Or should it be inside the hashfunction, so that it produces a valid formatted hash?
            pool.reserve(hash_functions.size());

            for (size_t i = 0; i < hash_functions.size(); i++) {
                pool.push_back(hash_functions[i](input));
            }

            for (size_t rep = 0; rep < num_tables; rep++) {
                // Concatenate the hashes
                hashType res;
                for (auto idx : indices[rep]) {
                    res.concatenate_hash(pool[idx], bits_per_function);
                }
                res >>= bits_to_cut;
                output.push_back(res);
            }

        }

        float icollision_probability(float p) const {
            return hash_family.icollision_probability(p);
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) const {
            return hash_family.collision_probability(similarity, num_bits);
        }

        // This assumes that hashes are independent, which is not true.
        // Therefore using a pool can result in recalls that are lower than expected.
        virtual float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t tables,
            uint_fast32_t max_tables,
            float kth_similarity
        ) const {
            float col_prob =
                this->concatenated_collision_probability(hash_length, kth_similarity);
            float last_prob =
                this->concatenated_collision_probability(hash_length+1, kth_similarity);
            return std::pow(1.0-col_prob, tables)*std::pow(1-last_prob, max_tables-tables);
        }

    };

    /// Describes a hash source which precomputes a pool of a given size.
    /// 
    /// Each hash is then constructed by sampling from this pool.
    /// This reduces the number of hashes that need to be computed, but produces hashes of lower quality.
    ///
    /// It is typically possible to choose a pool size which 
    /// performs better than independent hashing,
    /// but using independent hashes is a better default.
    template <typename T, typename hashType>
    struct HashPoolArgs : public HashSourceArgs<T, hashType> {
        /// Arguments for the hash family.
        typename T::Args args;
        /// The size of the pool in bits.
        unsigned int pool_size;

        constexpr HashPoolArgs(unsigned int pool_size)
          : pool_size(pool_size)
        {
        }

        HashPoolArgs(std::istream& in)
          : args(in)
        {
            in.read(reinterpret_cast<char*>(&pool_size), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            HashSourceType type = HashSourceType::Pool;
            out.write(reinterpret_cast<const char*>(&type), sizeof(HashSourceType));

            args.serialize(out);
            out.write(reinterpret_cast<const char*>(&pool_size), sizeof(unsigned int));
        }

        std::unique_ptr<HashSource<T, hashType>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int num_tables,
            unsigned int num_bits_per_function
        ) const {
            return std::make_unique<HashPool<T, hashType>> (
                desc,
                args,
                pool_size,
                num_tables,
                num_bits_per_function
            );
        }

        std::unique_ptr<HashSourceArgs<T, hashType>> copy() const {
            return std::make_unique<HashPoolArgs<T, hashType>>(*this);
        }

        uint64_t memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int /*num_tables*/,
            unsigned int /*num_bits*/
        ) const {
            typename T::Args args_copy(args);
            args_copy.set_no_preprocessing();
            auto bits = T(dataset, args_copy).bits_per_function();
            return sizeof(HashPool<T, hashType>)
                + pool_size/bits*args.memory_usage(dataset);
        }

        uint64_t function_memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int num_bits
        ) const {
            typename T::Args args_copy(args);
            args_copy.set_no_preprocessing();
            auto bits = T(dataset, args_copy).bits_per_function();
            return (num_bits+bits-1)/bits*sizeof(unsigned int);
        }

        std::unique_ptr<HashSource<T, hashType>> deserialize_source(std::istream& in) const {
            return std::make_unique<HashPool<T, hashType>>(in);
        }
    };
}
