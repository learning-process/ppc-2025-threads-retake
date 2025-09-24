#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
// #include "core/util/include/util.hpp"
#include "seq/strakhov_a_double_radix_merge/include/ops_seq.hpp"

namespace {
std::vector<double> RunMyTask(const std::vector<double> &input) {
  std::vector<double> out(input.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint64_t *>(const_cast<double *>(input.data())));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint64_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  strakhov_a_double_radix_merge_seq::DoubleRadixMergeSeq task(task_data_seq);
  EXPECT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  return out;
}
}  // namespace
TEST(strakhov_a_double_radix_merge, test_simple1) {
  std::vector<double> in{1.1, 2.2, 3.3, -4.4};
  std::vector<double> expected{-4.4, 1.1, 2.2, 3.3};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
