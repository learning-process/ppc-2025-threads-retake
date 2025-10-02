#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_connected_components_tbb {

struct Point {
  int x = 0;
  int y = 0;
  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
};

class ConnectedComponentsTbb : public ppc::core::Task {
 public:
  explicit ConnectedComponentsTbb(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_image_;
  std::vector<int> output_labels_;
  int width_ = 0;
  int height_ = 0;
  int components_count_ = 0;
};

}  // namespace dudchenko_o_connected_components_tbb