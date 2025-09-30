#pragma once

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

  void ParallelSort(std::vector<int>& data, std::size_t n, int available_threads);
  static void PerformParallelMerge(std::vector<int>& data, std::vector<std::pair<std::size_t, std::size_t>>& segments);
  void TwoThreadSort(std::vector<int>& data, std::size_t left_size, std::size_t n);
  void SequentialSort(std::vector<int>& data, std::size_t left_size, std::size_t n);

  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace ersoz_b_hoare_sort_simple_merge_stl
