#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ersoz_b_hoare_sort_simple_merge_seq {

class HoareSortSimpleMergeSequential : public ppc::core::Task {
 public:
  explicit HoareSortSimpleMergeSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void QuickSortHoare(std::vector<int>& a, long long l, long long r);
  static void MergeTwo(const std::vector<int>& src, std::pair<std::size_t, std::size_t> left,
                       std::pair<std::size_t, std::size_t> right, std::vector<int>& dst);

  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace ersoz_b_hoare_sort_simple_merge_seq
