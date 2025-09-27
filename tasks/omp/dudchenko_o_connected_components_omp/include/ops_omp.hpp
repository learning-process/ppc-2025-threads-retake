#pragma once

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

  void FirstPass(std::vector<int>& labels, std::vector<int>& parent) const;
  void SecondPass(std::vector<int>& labels, const std::vector<int>& parent) const;
  int FindRoot(std::vector<int>& parent, int x) const;
};

}  // namespace dudchenko_o_connected_components_omp