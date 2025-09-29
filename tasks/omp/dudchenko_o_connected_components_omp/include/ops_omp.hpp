#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_connected_components_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // Сохраняем структуры как в sequential версии
  struct ComponentLabels {
    std::vector<int> labels;
  };

  struct ParentStructure {
    std::vector<int32_t> parents;
  };

  std::vector<int> input_;
  std::vector<int> output_;
  int width_{};
  int height_{};

  void LabelComponents();
  void FirstPass(ComponentLabels& component_labels, ParentStructure& parent_structure);
  void SecondPass(ComponentLabels& component_labels, ParentStructure& parent_structure);
  void ProcessPixel(int x, int y, ComponentLabels& component_labels,
                    ParentStructure& parent_structure, int& local_next_label);
  void ProcessConnectedNeighbors(int left_label, int top_label, ComponentLabels& component_labels,
                                 ParentStructure& parent_structure, int index);
  void ProcessBlock(int start_y, int end_y, ComponentLabels& component_labels,
                    ParentStructure& parent_structure, int base_label);
  void ResolveBlockBoundaries(ComponentLabels& component_labels, ParentStructure& parent_structure);
  
  int FindRoot(ParentStructure& parent, int x);
  void UnionSets(ParentStructure& parent, int x, int y);
};

}  // namespace dudchenko_o_connected_components_omp