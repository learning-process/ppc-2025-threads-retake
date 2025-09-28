#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kalinin_d_simpson_method/include/ops_tbb.hpp"

namespace {

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& lower, const std::vector<double>& upper,
                                                  int segments_per_dim, int function_id, double* result_ptr) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* lower_ptr = const_cast<double*>(lower.data());
  auto* upper_ptr = const_cast<double*>(upper.data());

  static thread_local std::deque<std::array<int, 2>> k_params_storage;
  k_params_storage.emplace_back(std::array<int, 2>{segments_per_dim, function_id});
  int* params_ptr = k_params_storage.back().data();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_ptr));
  task_data->inputs_count.emplace_back(lower.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_ptr));
  task_data->inputs_count.emplace_back(upper.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(params_ptr));
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_ptr));
  task_data->outputs_count.emplace_back(1);
  return task_data;
}

}  // namespace

TEST(kalinin_d_simpson_method_tbb, one_dim_constant) {
  std::vector<double> a{0.0};
  std::vector<double> b{5.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 5.0, 1e-9);
}

TEST(kalinin_d_simpson_method_tbb, one_dim_linear) {
  std::vector<double> a{0.0};
  std::vector<double> b{1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.5, 1e-9);
}

TEST(kalinin_d_simpson_method_tbb, one_dim_quadratic) {
  std::vector<double> a{0.0};
  std::vector<double> b{1.0};
  int n = 100;
  double res = 0.0;

  auto task_data = MakeTaskData(a, b, n, 3, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0 / 3.0, 1e-9);
}

TEST(kalinin_d_simpson_method_tbb, two_dim_constant_rect) {
  std::vector<double> a{0.0, 0.0};
  std::vector<double> b{2.0, 3.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 6.0, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, two_dim_linear_sum_unit_square) {
  std::vector<double> a{0.0, 0.0};
  std::vector<double> b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, two_dim_product_unit_square) {
  std::vector<double> a{0.0, 0.0};
  std::vector<double> b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 2, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.25, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, two_dim_sum_squares_unit_square) {
  std::vector<double> a{0.0, 0.0};
  std::vector<double> b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 3, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 2.0 / 3.0, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, three_dim_constant_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0};
  std::vector<double> b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, three_dim_linear_sum_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0};
  std::vector<double> b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.5, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, three_dim_product_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0};
  std::vector<double> b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 2, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.125, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, two_dim_constant_mixed_bounds) {
  std::vector<double> a{1.0, 2.0};
  std::vector<double> b{3.0, 5.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 6.0, 1e-8);
}

TEST(kalinin_d_simpson_method_tbb, validation_odd_segments) {
  std::vector<double> a{0.0};
  std::vector<double> b{1.0};
  int n = 11;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(kalinin_d_simpson_method_tbb, four_dim_constant_hyperrectangle) {
  std::vector<double> a{0.0, 0.0, -1.0, 2.0};
  std::vector<double> b{1.0, 2.0, 1.0, 4.0};

  int n = 10;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_tbb::SimpsonNDTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 8.0, 1e-8);
}
