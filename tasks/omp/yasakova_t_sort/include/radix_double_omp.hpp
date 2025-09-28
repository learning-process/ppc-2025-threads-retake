#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_sort_omp {

inline bool is_nan(double x) { return std::isnan(x); }

inline uint64_t to_key(double x) {
  uint64_t b = std::bit_cast<uint64_t>(x);
  if ((b >> 63) == 0) return b ^ 0x8000'0000'0000'0000ull;
  return ~b;
}

inline double from_key(uint64_t k) {
  if ((k >> 63) != 0) {
    uint64_t b = k ^ 0x8000'0000'0000'0000ull;
    return std::bit_cast<double>(b);
  }
  uint64_t b = ~k;
  return std::bit_cast<double>(b);
}

class SortTaskOpenMP : public ppc::core::Task {
 public:
  explicit SortTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> input_;
  std::vector<double> output_;
};

void radix_sort_double_omp(std::vector<double>& a);

}  // namespace yasakova_t_sort_omp
