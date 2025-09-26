#pragma once

#include <vector>
#include <cstdint>

#include "core/task/include/task.hpp"

namespace shishkarev_a_radix_sort {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  
  void radixSort(std::vector<int>& arr);
  int getMax(const std::vector<int>& arr);
  void countSort(std::vector<int>& arr, int exp);
  void batcherOddEvenMerge(std::vector<int>& arr, int left, int right);
};

}  // namespace shishkarev_a_radix_sort