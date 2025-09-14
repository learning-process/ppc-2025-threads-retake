#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_block_gauss_seq {

class BlockGaussSequential : public ppc::core::Task {
 public:
  explicit BlockGaussSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), xres(0), yres(0) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  int xres, yres;
};

}  // namespace anikin_m_block_gauss_seq
