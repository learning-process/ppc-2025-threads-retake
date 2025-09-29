#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_sort_seq {

inline bool IsNan(double x) { return std::isnan(x); }

inline auto ToKey(double x) -> uint64_t {
  const auto bits = std::bit_cast<uint64_t>(x);
  if ((bits >> 63) == 0) {
    return bits ^ 0x8000'0000'0000'0000ULL;
  }
  return ~bits;
}

inline auto FromKey(uint64_t key) -> double {
  if ((key >> 63) != 0) {
    const auto bits = key ^ 0x8000'0000'0000'0000ULL;
    return std::bit_cast<double>(bits);
  }
  const auto bits = ~key;
  return std::bit_cast<double>(bits);
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

void RadixSortDoubleSeq(std::vector<double>& data);

}  // namespace yasakova_t_sort_seq
