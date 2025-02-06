#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/math.hpp"
#include "puffinn/similarity_measure/l2.hpp"

namespace puffinn{

    class L2HashFunction{
    AlignedStorage<RealVectorFormat> hash_vec;
    unsigned int dimensions, bits;    
    float r,b;
    unsigned int ub;
    
    public:
        L2HashFunction(DatasetDescription<RealVectorFormat> dataset, unsigned int bits ,float r, float b)
            : hash_vec(allocate_storage<RealVectorFormat>(bits, dataset.storage_len)),
            dimensions(dataset.storage_len),
            bits(bits),
            r(r),
            b(b),
            ub(std::pow(2,bits) - 1)
        {
            auto vec = RealVectorFormat::generate_random(dataset.args);
            RealVectorFormat::store(vec, hash_vec.get(), dataset);
        }

        L2HashFunction(std::istream& in){
            in.read(reinterpret_cast<char*>(&dimensions), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&ub), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&r), sizeof(float));
            in.read(reinterpret_cast<char*>(&b), sizeof(float));
            hash_vec = allocate_storage<RealVectorFormat>(bits, dimensions);
            in.read(
                reinterpret_cast<char*>(hash_vec.get()),
                dimensions*sizeof(typename RealVectorFormat::Type)); 
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&dimensions), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&ub), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&b), sizeof(float));
            out.write(reinterpret_cast<const char*>(&r), sizeof(float));
        
            out.write(
                reinterpret_cast<const char*>(hash_vec.get()),
                dimensions*sizeof(typename RealVectorFormat::Type));
        }

        // Hash the given vector. negative values are handled by pushing everything into the positive range (at a cost of factor 2 of bits) 
        uint64_t operator()(const float* const vec) const {
            auto dot = dot_product(hash_vec.get(), vec, dimensions); //could be just std::inner_product
            uint64_t bucket = std::floor((dot + b) / r); 
            return (bucket > ub)? ub : bucket; //every bucket outside the allowed bits are pushed into the limit_buckets. 
        }    

        float getR(){
            return r;
        } 

        unsigned int getBits(){return bits;}

    };

    struct L2HashArgs {
        L2HashArgs() = default;

        L2HashArgs(std::istream&) {}

        void serialize(std::ostream&) const {}

        uint64_t memory_usage(DatasetDescription<RealVectorFormat> dataset) const {
            return sizeof(L2HashFunction) + dataset.storage_len*sizeof(RealVectorFormat::Type);
        }

        void set_no_preprocessing() {}
    };

class L2Hash{
    public: 
        using Args = L2HashArgs;
        using Sim = L2Similarity;
        using Function = L2HashFunction;

    private: 
        DatasetDescription<RealVectorFormat> dataset;

    public:
        L2Hash(DatasetDescription<RealVectorFormat> dataset, Args): 
        dataset(dataset){
        }

        L2Hash(std::istream& in): dataset(in)
        {
        }

        void serialize(std::ostream& out) const{
            dataset.serialize(out);
        }

        L2HashFunction sample(){
            float r_tmp = 4.0;
            std::normal_distribution<float> normal_distribution(0.0, r_tmp);
            auto& generator = get_default_random_generator();
            float b_tmp = normal_distribution(generator);

            return L2HashFunction(dataset, BITS_PER_FUNCTION, r_tmp, b_tmp); // see bits_per_function below;
        }


        unsigned int bits_per_function() {
            // return L2HashFunction.getBits(); //should be something like this, but current only a static value is allowed.
            return BITS_PER_FUNCTION;
        }

        //Similarity is know the value c, which is the distance/c; "distance" is the hashcode distance i.e. an integer.      
        //Currently assumes that r is the same across all concatenations and that num_bits is the number of concatenations
    
        float collision_probability(float c_tmp, int_fast8_t num_bits) const {
            float c = c_tmp/4;
            if(c < 0.001f) return 1.0f;
            return std::erf(1 / std::sqrt(2) / c) - c * std::sqrt(2 / M_PI) * (1 - std::exp(-0.5 / (c * c)));
        }
    
};


}
