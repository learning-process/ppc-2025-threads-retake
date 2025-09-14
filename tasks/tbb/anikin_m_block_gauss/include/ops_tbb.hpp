#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_block_gauss_tbb {

class BlockGaussTBB : public ppc::core::Task {
 public:
  explicit BlockGaussTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  int xres_, yres_;
};

}  // namespace anikin_m_block_gauss_tbb
