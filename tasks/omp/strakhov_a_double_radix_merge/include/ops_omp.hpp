#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace strakhov_a_double_radix_merge_omp {

class DoubleRadixMergeOmp : public ppc::core::Task {
 public:
  explicit DoubleRadixMergeOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  static inline void MergeSorted(const uint64_t *input, size_t left_start, size_t left_end, size_t right_start,
                                 size_t right_end, uint64_t *output, size_t output_start)
};

}  // namespace strakhov_a_double_radix_merge_omp