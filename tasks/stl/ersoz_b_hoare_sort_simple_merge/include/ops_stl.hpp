#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ersoz_b_hoare_sort_simple_merge_stl {

struct Segment {
  std::size_t begin{};
  std::size_t end{};
};

class HoareSortSimpleMergeSTL : public ppc::core::Task {
 public:
  explicit HoareSortSimpleMergeSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static long long Partition(std::vector<int>& a, long long l, long long r);
  static void QuickSortHoare(std::vector<int>& a, long long l, long long r);
  static void MergeTwo(const std::vector<int>& src, Segment left, Segment right, std::vector<int>& dst);
  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace ersoz_b_hoare_sort_simple_merge_stl
