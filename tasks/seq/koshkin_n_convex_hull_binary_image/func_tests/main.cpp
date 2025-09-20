#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/koshkin_n_convex_hull_binary_image/include/ops_seq.hpp"

TEST(koshkin_n_convex_hull_binary_image_seq, small_4x4) {
  int height = 4;
  int width = 4;

  std::vector<int> image = {0, 0, 0, 0,

                            0, 1, 1, 0,

                            0, 1, 1, 0,

                            0, 0, 0, 0};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(out, exp);
}

TEST(koshkin_n_convex_hull_binary_image_seq, small_5x5) {
  int height = 5;
  int width = 5;

  std::vector<int> image = {1, 1, 1, 1, 1,

                            1, 0, 0, 0, 1,

                            1, 0, 0, 0, 1,

                            1, 0, 0, 0, 1,

                            1, 1, 1, 1, 1};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{0, 0}, {0, 4}, {4, 0}, {4, 4}};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(out, exp);
}

TEST(koshkin_n_convex_hull_binary_image_seq, medium_4x8) {
  int height = 4;
  int width = 8;

  std::vector<int> image = {0, 0, 0, 0, 0, 0, 0, 0,

                            0, 1, 1, 1, 0, 0, 0, 0,

                            0, 1, 0, 1, 1, 1, 0, 0,

                            0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{1, 1}, {1, 2}, {3, 1}, {5, 2}};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(out, exp);
}

TEST(koshkin_n_convex_hull_binary_image_seq, collinear_horizontal_line) {
  int height = 1;
  int width = 5;

  std::vector<int> image = {1, 1, 1, 1, 1};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{0, 0}, {4, 0}};

  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(out, exp);
}