#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/kavtorev_d_batcher_sort/include/ops_tbb.hpp"

using kavtorev_d_batcher_sort_tbb::RadixBatcherSortTBB;

namespace {
std::vector<double> RunTask(const std::vector<double>& input) {
  std::vector<double> out(input.size());
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input.data())));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  RadixBatcherSortTBB task(task_data_tbb);
  EXPECT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  return out;
}
}  // namespace

TEST(kavtorev_d_batcher_sort_tbb, handles_empty) {
  std::vector<double> in;
  auto out = RunTask(in);
  EXPECT_TRUE(out.empty());
}

TEST(kavtorev_d_batcher_sort_tbb, handles_single) {
  std::vector<double> in{42.0};
  auto out = RunTask(in);
  EXPECT_EQ(out.size(), static_cast<size_t>(1));
  EXPECT_DOUBLE_EQ(out[0], 42.0);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_already_sorted) {
  std::vector<double> in{-5.0, -1.0, 0.0, 0.5, 2.0};
  auto out = RunTask(in);
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_reverse_sorted) {
  std::vector<double> in{9.0, 4.0, 1.0, -2.0, -7.0};
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_with_duplicates) {
  std::vector<double> in{3.0, 2.0, 3.0, -1.0, 2.0, 3.0};
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_negatives_and_positives) {
  std::vector<double> in{-0.1, -100.0, 50.5, 0.0, 3.14};
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_with_infinities) {
  std::vector<double> in{-std::numeric_limits<double>::infinity(), -1.0, 0.0, 1.0,
                         std::numeric_limits<double>::infinity()};
  auto out = RunTask(in);
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_with_nan_moved_last) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> in{3.0, nan, -1.0, 2.0};
  auto out = RunTask(in);
  std::vector<double> ref{-1.0, 2.0, 3.0, nan};
  for (size_t i = 0; i + 1 < out.size() - 0; ++i) {
    if (!std::isnan(out[i + 1])) {
      EXPECT_LE(out[i], out[i + 1]);
    }
  }
  EXPECT_TRUE(std::isnan(out.back()));
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_random_small) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> d(-1000.0, 1000.0);
  std::vector<double> in(101);
  for (auto& v : in) {
    v = d(gen);
  }
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  ASSERT_EQ(out.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], in[i]);
  }
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_power_of_two) {
  std::vector<double> in{8, 7, 6, 5, 4, 3, 2, 1};
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, in);
}

TEST(kavtorev_d_batcher_sort_tbb, sorts_large_random) {
  std::mt19937 gen(7);
  std::normal_distribution<double> d(0.0, 100.0);
  std::vector<double> in(500);
  for (auto& v : in) {
    v = d(gen);
  }
  auto out = RunTask(in);
  auto ref = in;
  std::sort(ref.begin(), ref.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, ref);
}

TEST(kavtorev_d_batcher_sort_tbb, file_driven_small) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/example/data/test.txt"));
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();
  const size_t count = std::stoi(line);
  std::vector<double> in(count);
  for (size_t i = 0; i < count; ++i) {
    in[i] = static_cast<double>(count - i);
  }
  auto out = RunTask(in);
  std::sort(in.begin(), in.end());  // NOLINT(modernize-use-ranges)
  EXPECT_EQ(out, in);
}