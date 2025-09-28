#pragma once

#include <cstdint>
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
  using ComponentLabels = std::vector<int>;
  using ParentStructure = std::vector<int32_t>;

  std::vector<int> input_;
  std::vector<int> output_;
  int width_;
  int height_;

  void LabelComponents();
  void FirstPass(ComponentLabels& component_labels, ParentStructure& parent_structure);
  static void SecondPass(ComponentLabels& component_labels, const ParentStructure& parent_structure);
  static int FindRoot(ParentStructure& parent, int x);
  static void UnionSets(ParentStructure& parent, int x, int y);
};

}  // namespace dudchenko_o_connected_components