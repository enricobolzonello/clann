#pragma once

#include "puffinn/format/real_vector.hpp"

namespace puffinn {
    class L2Hash;

    struct L2Similarity {
        using Format = RealVectorFormat;
        using DefaultHash = L2Hash;
        using DefaultSketch = L2Hash;

        static float compute_similarity(float* lhs, float* rhs, DatasetDescription<Format> desc) {
            auto dist = l2_distance_float(lhs, rhs, desc.args);
            // Convert to a similarity between 0 and 1,
            // which is needed to calculate collision probabilities.
            return 1.0/(dist+1.0);
            // return dist;
        }
    };
}

#include "puffinn/hash/L2hash.hpp"
