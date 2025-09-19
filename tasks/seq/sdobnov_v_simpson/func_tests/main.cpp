#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sdobnov_v_simpson/include/ops_seq.hpp"

TEST(sdobnov_v_simpson, test_2d_integral_x_squared_plus_y_squared) {
  const int dimensions = 2;
  const double lower_bounds[2] = {0.0, 0.0};
  const double upper_bounds[2] = {1.0, 1.0};
  const int n_points[2] = {100, 100};
  double result = 0.0;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(2 * sizeof(int));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(TestFunction2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = 2.0 / 3.0;
  const double tolerance = 1e-4;

  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(sdobnov_v_simpson, validation_fails_with_no_inputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson, validation_fails_with_insufficient_inputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = 2;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  task_data->inputs_count.emplace_back(sizeof(int));

  double bounds[2] = {0.0, 0.0};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson, validation_fails_with_invalid_dimensions) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = -1;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  task_data->inputs_count.emplace_back(sizeof(int));

  double bounds[1] = {0.0};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(sizeof(double));

  int points[1] = {10};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(points));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&Gaussian2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson, validation_fails_with_wrong_array_sizes) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = 2;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  task_data->inputs_count.emplace_back(sizeof(int));

  double bounds[1] = {0.0};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds));
  task_data->inputs_count.emplace_back(sizeof(double));

  int points[1] = {10};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(points));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&Gaussian2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson, preprocessing_fails_with_null_function) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = 1;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  task_data->inputs_count.emplace_back(sizeof(int));

  double lower_bound = 0.0;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
  task_data->inputs_count.emplace_back(sizeof(double));

  double upper_bound = 1.0;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
  task_data->inputs_count.emplace_back(sizeof(double));

  int points = 10;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&points));
  task_data->inputs_count.emplace_back(sizeof(int));

  sdobnov_v_simpson::Func null_func = nullptr;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(null_func));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_TRUE(test_task.Validation());
  EXPECT_FALSE(test_task.PreProcessing());
}

TEST(sdobnov_v_simpson, preprocessing_fails_with_invalid_points) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = 1;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  task_data->inputs_count.emplace_back(sizeof(int));

  double lower_bound = 0.0;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
  task_data->inputs_count.emplace_back(sizeof(double));

  double upper_bound = 1.0;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
  task_data->inputs_count.emplace_back(sizeof(double));

  int points = 0;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&points));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&LinearFunction1d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);
  EXPECT_TRUE(test_task.Validation());
  EXPECT_FALSE(test_task.PreProcessing());
}

TEST(sdobnov_v_simpson, test_1d_integral_linear_function) {
  const int dimensions = 1;
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n_points = 100;

  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&lower_bound)));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&upper_bound)));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n_points)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&LinearFunction1d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = 0.5;
  const double tolerance = 1e-6;

  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(sdobnov_v_simpson, test_gaussian_2d_integral) {
  const int dimensions = 2;
  const double lower_bounds[2] = {-1.0, -1.0};
  const double upper_bounds[2] = {1.0, 1.0};
  const int n_points[2] = {100, 100};

  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(2 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&Gaussian2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = 2.230;
  const double tolerance = 1e-2;

  EXPECT_NEAR(result, expected_result, tolerance);
}

// ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ
TEST(sdobnov_v_simpson, test_single_point_integration) {
  const int dimensions = 1;
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n_points = 1;

  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&lower_bound)));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&upper_bound)));
  task_data->inputs_count.emplace_back(sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n_points)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&LinearFunction1d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson::SimpsonIntegralSequential test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = 0.5;
  const double tolerance = 1e-4;

  EXPECT_NEAR(result, expected_result, tolerance);
}