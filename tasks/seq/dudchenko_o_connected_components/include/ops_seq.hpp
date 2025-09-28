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
  // Объявляем типы для устранения предупреждения о легко переставляемых параметрах
  using LabelVector = std::vector<int>;
  using ParentVector = std::vector<int32_t>;

  std::vector<int> input_;
  std::vector<int> output_;
  int width_;
  int height_;

  void LabelComponents();
  void FirstPass(LabelVector& component_labels, ParentVector& parent_structure);
  static void SecondPass(LabelVector& component_labels, const ParentVector& parent_structure);
  static int FindRoot(ParentVector& parent, int x);
  static void UnionSets(ParentVector& parent, int x, int y);
};

}  // namespace dudchenko_o_connected_components