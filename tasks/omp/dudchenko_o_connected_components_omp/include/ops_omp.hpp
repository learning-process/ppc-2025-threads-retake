#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_connected_components_omp {

class ConnectedComponentsOmp : public ppc::core::Task {
 public:
  explicit ConnectedComponentsOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_image_;
  std::vector<int> output_labels_;
  int width_ = 0;
  int height_ = 0;
  int components_count_ = 0;

  void ProcessPixel(int x, int y, std::vector<int>& pixel_labels, std::vector<int>& union_find_parent, int& next_label);
  static void CreateNewComponent(std::vector<int>& pixel_labels, std::vector<int>& union_find_parent, int& next_label,
                                 size_t idx);
  static void HandleBothNeighbors(std::vector<int>& pixel_labels, std::vector<int>& union_find_parent, size_t idx,
  static void UnionComponents(std::vector<int>& union_find_parent, int min_label_val, int max_label_val,
                              int root_min_val, int root_max_val);
  void ResolveLabels(std::vector<int>& labels, const std::vector<int>& parent);
  void CompactLabels(const std::vector<int>& labels);
  [[nodiscard]] static int FindRoot(const std::vector<int>& parent, int x);
};

}  // namespace dudchenko_o_connected_components_omp