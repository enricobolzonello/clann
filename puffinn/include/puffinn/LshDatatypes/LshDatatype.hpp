#pragma once

#include <stdint.h>
#include <puffinn/typedefs.hpp>

struct LshDatatype_DECL{
    
    virtual void concatenate_hashes(const std::vector<unsigned int>& indices, const uint64_t* hashes,const uint_fast8_t& bits_per_function) = 0;
    virtual void concatenate_hash(const uint64_t& hash,const uint_fast8_t& bits_per_function) = 0;
    virtual void operator<<= (int bits) = 0;    
    virtual void pop_hash(unsigned int bits) = 0; 

};


template <typename dataType>
struct HammingType: public LshDatatype_DECL //How should I set the scope? private/protected/public
{
public:

    dataType value;
    
    HammingType(){
        this->value = 0;
    }

    HammingType(dataType value){
        this->value = value;
    }

    void concatenate_hashes(
        const std::vector<unsigned int>& indices, 
        const uint64_t* hashes,
        const uint_fast8_t& bits_per_function
    ) override { 
        for (auto idx : indices) {
            this->value <<= bits_per_function;
            this->value |= hashes[idx];
        }
    }

    void concatenate_hash(
        const uint64_t& hash,
        const uint_fast8_t& bits_per_func
    ) override {
        this->value <<= bits_per_func;
        this->value |= hash; 
    }

    HammingType intersperse_zero() const {
        dataType mask = 1;
        dataType shift = 0;
        dataType res = 0;
        for (unsigned i=0; i < sizeof(dataType)*8/2; i++) {
            res |= (this->value & mask) << shift;
            mask <<= 1;
            shift++;
        }
        return res;
    }
    //This is a hacky solution for now, this works because the function is only ever called for a prefix_mask which starts as all 1's
    void pop_hash(unsigned int bits) {
        this->value = this->value << bits;  
    } 
    
    bool operator!=(const HammingType& other) const{
        return this->value != other.getValue();
    }

    bool prefix_eq(HammingType<dataType> other, HammingType<dataType> mask) const{
        return this->value == (other & mask); 
    }

    HammingType interleave(const HammingType<dataType>& other) const {
        return this->value | other.getValue();
    }

    dataType operator^(HammingType<dataType> other) const {
        return this->value ^ other.getValue();
    }

    dataType operator>>(unsigned int shift_amount) const {
        return this->value >> shift_amount;
    }

    void operator>>=(unsigned int shift_amount) {
        this->value >>= shift_amount;
    }

    void operator|=(HammingType<dataType> mask){
        this->value |= mask.getValue();
    }

    void operator<<= (int bits) override {
        this->value = this->value << bits;
    }

    dataType getValue() const{
        return this->value;
    }

    dataType getBPF() const{
        return this->bits_per_function;
    }

    bool operator< (HammingType<dataType> const& other) const {
        return this->value < other.getValue();
    }


    // bool prefix_eq(LshDatatype_DECL<dataType> const &other, uint32_t prefix_length) const {
    //     return 0;
    // }

    dataType operator& (HammingType<dataType> const& other) const {
        return this->value & other.getValue();
    }

};

