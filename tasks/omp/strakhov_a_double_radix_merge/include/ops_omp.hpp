#pragma once

#include <cstddef>
#include <cstdint>
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
  struct Range;
  inline void FloatToNormalized(std::vector<uint64_t> &input, unsigned size);
  inline void NormalizedToFloat(std::vector<uint64_t> &input, unsigned size);
  static inline void RadixGrandSort(size_t n, std::vector<uint64_t> &input, size_t chunk_size, int type_length);
  std::vector<double> input_, output_;
  static inline void MergeSorted(const uint64_t *input, Range left, Range right, uint64_t *output, size_t output_start);
};

}  // namespace strakhov_a_double_radix_merge_omp