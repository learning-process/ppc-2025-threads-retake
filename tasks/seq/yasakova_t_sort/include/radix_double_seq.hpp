#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_sort_seq {

inline bool is_nan(double x) { return std::isnan(x); }

// double -> lexicographically comparable key (IEEE-754 order)
inline uint64_t to_key(double x) {
    uint64_t b = std::bit_cast<uint64_t>(x);
    if ((b >> 63) == 0) return b ^ 0x8000'0000'0000'0000ull;
    return ~b;
}
// key -> double (inverse transform)
inline double from_key(uint64_t k) {
    if ((k >> 63) != 0) {
        uint64_t b = k ^ 0x8000'0000'0000'0000ull;
        return std::bit_cast<double>(b);
    } else {
        uint64_t b = ~k;
        return std::bit_cast<double>(b);
    }
}

class SortTaskSequential : public ppc::core::Task {
 public:
    explicit SortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

 private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;

    std::vector<double> input_;
    std::vector<double> output_;
};

// LSD-radix sort over 8-bit passes
void radix_sort_double_seq(std::vector<double>& a);

} // namespace yasakova_t_sort_seq
