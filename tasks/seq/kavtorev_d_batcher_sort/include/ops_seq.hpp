#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_batcher_sort_seq {

class RadixBatcherSortSequential : public ppc::core::Task {
 public:
  explicit RadixBatcherSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;

  static inline uint64_t ToOrderedUint64(double value);
  static inline double FromOrderedUint64(uint64_t key);
  static void LsdRadixSortUint64(std::vector<uint64_t>& data);

  static void OddEvenMerge(std::vector<double>& a, int left, int size, int stride);
  static void OddEvenMergeSort(std::vector<double>& a, int left, int size);
};

}  // namespace kavtorev_d_batcher_sort_seq