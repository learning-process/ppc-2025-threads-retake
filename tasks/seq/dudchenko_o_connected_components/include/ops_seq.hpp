#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_connected_components {

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
  int width_;
  int height_;

  void LabelComponents();
  void FirstPass(std::vector<int>& labels, std::vector<int>& parent);
  void SecondPass(std::vector<int>& labels, const std::vector<int>& parent);
  int FindRoot(std::vector<int>& parent, int x);
  void UnionSets(std::vector<int>& parent, int x, int y);
};

}  // namespace dudchenko_o_connected_components