#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_graham_tbb {

class GrahamTbb : public ppc::core::Task {
 public:
  explicit GrahamTbb(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<float> input_X_, input_Y_;
  std::vector<float> output_X_, output_Y_;
  static std::pair<float, float> Minus(std::pair<float, float> a, std::pair<float, float> b);
  static float Mul(std::pair<float, float> a, std::pair<float, float> b);
  static std::pair<float, float> GetMinPoint(const std::vector<std::pair<float, float>> &points);
  static void InitData(std::vector<std::vector<std::pair<float, float>>> &data, int threads, size_t temp_size,
                       const std::vector<std::pair<float, float>> &points);
  static void WhileLoop(const std::pair<float, float> &p, std::vector<std::pair<float, float>> &hull);
};

class GrahamSeq : public ppc::core::Task {
 public:
  explicit GrahamSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<float> input_X_, input_Y_;
  std::vector<float> output_X_, output_Y_;
  static std::pair<float, float> Minus(std::pair<float, float> a, std::pair<float, float> b);
  static float Mul(std::pair<float, float> a, std::pair<float, float> b);
};
}  // namespace leontev_n_graham_tbb
