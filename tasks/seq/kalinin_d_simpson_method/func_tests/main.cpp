#include <gtest/gtest.h>

#include <array>
#include <cmath>
#
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kalinin_d_simpson_method/include/ops_seq.hpp"

namespace {

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& lower, const std::vector<double>& upper,
                                                  int segments_per_dim, int function_id, double* result_ptr) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* lower_ptr = const_cast<double*>(lower.data());
  auto* upper_ptr = const_cast<double*>(upper.data());
  // Persist params storage to avoid dangling pointer and keep addresses stable
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

// 1D: f(x)=1 on [a,b] => integral = b-a
TEST(kalinin_d_simpson_method_seq, one_dim_constant) {
  std::vector<double> a{0.0}, b{5.0};
  int n = 100;  // even
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 5.0, 1e-9);
}

// 1D: f(x)=x on [0,1] => 1/2
TEST(kalinin_d_simpson_method_seq, one_dim_linear) {
  std::vector<double> a{0.0}, b{1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.5, 1e-9);
}

// 1D: f(x)=x^2 on [0,1] => 1/3
TEST(kalinin_d_simpson_method_seq, one_dim_quadratic) {
  std::vector<double> a{0.0}, b{1.0};
  int n = 100;
  double res = 0.0;
  // function_id 3 is sum of squares; in 1D equals x^2
  auto task_data = MakeTaskData(a, b, n, 3, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0 / 3.0, 1e-9);
}

// 2D: f(x,y)=1 on [0,2]x[0,3] => area=6
TEST(kalinin_d_simpson_method_seq, two_dim_constant_rect) {
  std::vector<double> a{0.0, 0.0}, b{2.0, 3.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 6.0, 1e-8);
}

// 2D: f(x,y)=x+y on [0,1]^2 => 1/2 + 1/2 = 1
TEST(kalinin_d_simpson_method_seq, two_dim_linear_sum_unit_square) {
  std::vector<double> a{0.0, 0.0}, b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0, 1e-8);
}

// 2D: f(x,y)=x*y on [0,1]^2 => (1/2)*(1/2)=1/4
TEST(kalinin_d_simpson_method_seq, two_dim_product_unit_square) {
  std::vector<double> a{0.0, 0.0}, b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 2, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.25, 1e-8);
}

// 2D: f(x,y)=x^2+y^2 on [0,1]^2 => 2/3
TEST(kalinin_d_simpson_method_seq, two_dim_sum_squares_unit_square) {
  std::vector<double> a{0.0, 0.0}, b{1.0, 1.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 3, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 2.0 / 3.0, 1e-8);
}

// 3D: f=1 on [0,1]^3 => 1
TEST(kalinin_d_simpson_method_seq, three_dim_constant_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0}, b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.0, 1e-8);
}

// 3D: f=x+y+z on [0,1]^3 => 3 * 1/2 = 3/2
TEST(kalinin_d_simpson_method_seq, three_dim_linear_sum_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0}, b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 1.5, 1e-8);
}

// 3D: f=x*y*z on [0,1]^3 => (1/2)^3 = 1/8
TEST(kalinin_d_simpson_method_seq, three_dim_product_unit_cube) {
  std::vector<double> a{0.0, 0.0, 0.0}, b{1.0, 1.0, 1.0};
  int n = 20;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 2, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 0.125, 1e-8);
}

// Mixed bounds: f=1 on [1,3]x[2,5] => (2)*(3)=6
TEST(kalinin_d_simpson_method_seq, two_dim_constant_mixed_bounds) {
  std::vector<double> a{1.0, 2.0}, b{3.0, 5.0};
  int n = 100;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 6.0, 1e-8);
}

// Check validation: odd segments should fail
TEST(kalinin_d_simpson_method_seq, validation_odd_segments) {
  std::vector<double> a{0.0}, b{1.0};
  int n = 11;  // odd -> invalid
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

// Higher accuracy check by increasing segments
// TEST(kalinin_d_simpson_method_seq, accuracy_improves_with_segments) {
//   std::vector<double> a{0.0}, b{1.0};
//   double res_coarse = 0.0;
//   double res_fine = 0.0;

//   {
//     auto td = MakeTaskData(a, b, 10, 3, &res_coarse);
//     kalinin_d_simpson_method_seq::SimpsonNDSequential t(td);
//     ASSERT_TRUE(t.Validation());
//     t.PreProcessing();
//     t.Run();
//     t.PostProcessing();
//   }

//   {
//     auto td = MakeTaskData(a, b, 100, 3, &res_fine);
//     kalinin_d_simpson_method_seq::SimpsonNDSequential t(td);
//     ASSERT_TRUE(t.Validation());
//     t.PreProcessing();
//     t.Run();
//     t.PostProcessing();
//   }

//   double exact = 1.0 / 3.0;
//   EXPECT_GT(std::abs(exact - res_coarse), std::abs(exact - res_fine));
// }

// Dimension 4 constant: volume of hyper-rectangle
TEST(kalinin_d_simpson_method_seq, four_dim_constant_hyperrectangle) {
  std::vector<double> a{0.0, 0.0, -1.0, 2.0};
  std::vector<double> b{1.0, 2.0, 1.0, 4.0};
  // volume = 1*2*2*2 = 8
  int n = 10;
  double res = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &res);
  kalinin_d_simpson_method_seq::SimpsonNDSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(res, 8.0, 1e-8);
}
