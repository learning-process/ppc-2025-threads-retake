#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_graham_seq {

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
  std::pair<float, float> minus(std::pair<float, float> a, std::pair<float, float> b);
  float mul(std::pair<float, float> a, std::pair<float, float> b);
};

}  // namespace leontev_n_graham_seq