#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_n_convex_hull_binary_image_seq {
using Pt = std::pair<int, int>;
class ConvexHullBinaryImage : public ppc::core::Task {
 public:
  explicit ConvexHullBinaryImage(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static long long Cross(const Pt& a, const Pt& b, const Pt& c);

  static long long Dist2(const Pt& a, const Pt& b);

  void FindPoints();
  std::vector<int> input;
  std::vector<Pt> points;
  std::vector<Pt> output;
  int width{}, height{};
};

}  // namespace koshkin_n_convex_hull_binary_image_seq