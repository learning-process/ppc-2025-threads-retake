#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vragov_i_gaussian_filter_vertical/include/filter.hpp"

// Helper to create a simple vertical test image
template <typename T>
std::vector<T> make_vertical_gradient(int x, int y) {
  std::vector<T> img(x * y);
  for (int i = 0; i < x; ++i)
    for (int j = 0; j < y; ++j) img[i * y + j] = j * 10;
  return img;
}

TEST(vragov_i_gaussian_filter_vertical_seq, identity_on_constant_image) {
  constexpr int x = 5, y = 5;
  std::vector<int> in(x * y, 42);
  std::vector<int> out(x * y, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(x);
  task_data->inputs_count.emplace_back(y);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (int v : out) {
    EXPECT_NEAR(v, 27, 2);
  }
}

TEST(vragov_i_gaussian_filter_vertical_seq, vertical_gradient_blur) {
  constexpr int x = 3, y = 7;
  auto in = make_vertical_gradient<int>(x, y);
  std::vector<int> out(x * y, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(x);
  task_data->inputs_count.emplace_back(y);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  // Check that the output is blurred vertically (middle pixel is average of neighbors)
  for (int i = 0; i < x; ++i) {
    for (int j = 1; j < y - 1; ++j) {
      int idx = i * y + j;
      int expected = static_cast<int>(std::round((in[idx - 1] * 0.015 + in[idx] * 0.8 + in[idx + 1] * 0.015) /
                                                 (std::sqrt(2.0 * std::acos(-1.0)) * 0.5)));
      EXPECT_NEAR(out[idx], expected, 2);
    }
  }
}

TEST(vragov_i_gaussian_filter_vertical_seq, handles_empty_image) {
  std::vector<int> in;
  std::vector<int> out;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(0);
  task_data->inputs_count.emplace_back(0);
  task_data->inputs_count.emplace_back(0);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(0);

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_TRUE(out.empty());
}

TEST(vragov_i_gaussian_filter_vertical_seq, invalid_dimensions_fail_validation) {
  constexpr int x = 4, y = 5;
  // Only 10 elements, but x*y = 20
  std::vector<int> in(10, 1);
  std::vector<int> out(x * y, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(x);
  task_data->inputs_count.emplace_back(y);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(vragov_i_gaussian_filter_vertical_seq, single_element_image) {
  constexpr int x = 1, y = 1;
  std::vector<int> in = {123};
  std::vector<int> out(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(x);
  task_data->inputs_count.emplace_back(y);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(out[0], 79, 2);
}

TEST(vragov_i_gaussian_filter_vertical_seq, random_image_blur_average) {
  constexpr int x = 20, y = 20;
  std::vector<int> in(x * y);
  std::vector<int> out(x * y, 0);

  // Fill with random values in [0, 255]
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& v : in) v = dist(gen);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(x);
  task_data->inputs_count.emplace_back(y);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_seq::GaussianFilterTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  // Compute average of input and output
  double avg_in = 0.0, avg_out = 0.0;
  for (int v : in) avg_in += v;
  for (int v : out) avg_out += v;
  avg_in /= in.size();
  avg_out /= out.size();

  // Output average should be about 0.64 of input average (+-0.05 margin)
  EXPECT_NEAR(avg_out, avg_in * 0.64, avg_in * 0.05);
}