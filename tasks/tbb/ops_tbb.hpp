#pragma once

#include <mpi.h>
#include <string>

#include "core/task/include/task.hpp"

namespace bychikhina_k_ring_tbb {

class RingTopology {
public:
    RingTopology(int world_size, int world_rank);
    void sendMessage(int source, int dest, const std::string& msg);
private:
    int left;
    int right;
    int world_size;
    int world_rank;
};

}  // namespace bychikhina_k_ring_tbb
