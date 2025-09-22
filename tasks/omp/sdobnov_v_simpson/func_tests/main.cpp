#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/sdobnov_v_simpson/include/ops_omp.hpp"

TEST(sdobnov_v_simpson_omp, validation_fails_with_no_inputs) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson_omp, validation_fails_with_insufficient_inputs) {
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

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson_omp, validation_fails_with_invalid_dimensions) {
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

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Polynomial3d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson_omp, validation_fails_with_wrong_array_sizes) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  int dimensions = 3;
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

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Polynomial3d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(sdobnov_v_simpson_omp, preprocessing_fails_with_null_function) {
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

  sdobnov_v_simpson_omp::Func null_func = nullptr;
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(null_func));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_TRUE(test_task.Validation());
  EXPECT_FALSE(test_task.PreProcessing());
}

TEST(sdobnov_v_simpson_omp, preprocessing_fails_with_invalid_points) {
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

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Polynomial3d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);
  EXPECT_TRUE(test_task.Validation());
  EXPECT_FALSE(test_task.PreProcessing());
}

TEST(sdobnov_v_simpson_omp, test_polynomial_3d_integral) {
  const int dimensions = 3;
  const double lower_bounds[3] = {0.0, 0.0, 0.0};
  const double upper_bounds[3] = {1.0, 1.0, 1.0};
  const int n_points[3] = {50, 50, 50};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(3 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(3 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(3 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Polynomial3d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = 1.5;
  const double tolerance = 1e-2;

  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(sdobnov_v_simpson_omp, test_trigonometric_4d_integral) {
  const int dimensions = 4;
  const double lower_bounds[4] = {0.0, 0.0, 0.0, 0.0};
  const double upper_bounds[4] = {M_PI / 2, M_PI / 2, M_PI / 2, M_PI / 2};
  const int n_points[4] = {40, 40, 40, 40};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(4 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(4 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(4 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Trigonometric4d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = (M_PI * M_PI * M_PI) / 2;
  const double tolerance = 1e-2;

  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(sdobnov_v_simpson_omp, test_mixed_5d_integral) {
  const int dimensions = 5;
  const double lower_bounds[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  const double upper_bounds[5] = {1.0, M_PI / 2, 1.0, M_PI / 2, 1.0};
  const int n_points[5] = {20, 20, 20, 20, 20};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(5 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(5 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(5 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson_omp::Mixed5d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  sdobnov_v_simpson_omp::SimpsonIntegralOmp test_task(task_data);

  ASSERT_TRUE(test_task.Validation());
  ASSERT_TRUE(test_task.PreProcessing());
  ASSERT_TRUE(test_task.Run());
  ASSERT_TRUE(test_task.PostProcessing());

  const double expected_result = ((5 * M_PI * M_PI) + (18 * M_PI)) / 24;
  const double tolerance = 1e-2;

  EXPECT_NEAR(result, expected_result, tolerance);
}