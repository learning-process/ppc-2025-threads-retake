#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/dudchenko_o_connected_components_omp/include/ops_omp.hpp"

using dudchenko_o_connected_components_omp::ConnectedComponentsOmp;

namespace {
std::vector<int> RunComponents(const std::vector<uint8_t>& img, int w, int h) {
  std::vector<int> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(const_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto task = std::make_shared<ConnectedComponentsOmp>(td);

  if (!task->Validation()) {
    ADD_FAILURE() << "Validation() returned false";
    return {};
  }
  if (!task->PreProcessing()) {
    ADD_FAILURE() << "PreProcessing() returned false";
    return {};
  }
  if (!task->Run()) {
    ADD_FAILURE() << "Run() returned false";
    return {};
  }
  if (!task->PostProcessing()) {
    ADD_FAILURE() << "PostProcessing() returned false";
    return {};
  }

  return out;
}
int CountUniqueComponents(const std::vector<int>& labels) {
  std::vector<int> unique_labels;
  for (int label : labels) {
    if (label > 0 && std::ranges::find(unique_labels, label) == unique_labels.end()) {
      unique_labels.push_back(label);
    }
  }
  return static_cast<int>(unique_labels.size());
}
}  // namespace

TEST(dudchenko_o_connected_components_omp, single_pixel) {
  const int w = 3;
  const int h = 3;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);  // Белый фон
  img[4] = 0;

  auto labels = RunComponents(img, w, h);

  EXPECT_EQ(labels[4], 1);
  EXPECT_EQ(CountUniqueComponents(labels), 1);
}

TEST(dudchenko_o_connected_components_omp, two_separate_components) {
  const int w = 5;
  const int h = 5;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);

  img[6] = 0;
  img[18] = 0;

  auto labels = RunComponents(img, w, h);

  int label1 = labels[6];
  int label2 = labels[18];
  EXPECT_GT(label1, 0);
  EXPECT_GT(label2, 0);
  EXPECT_NE(label1, label2);
  EXPECT_EQ(CountUniqueComponents(labels), 2);
}

TEST(dudchenko_o_connected_components_omp, connected_component) {
  const int w = 4;
  const int h = 4;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);

  img[5] = 0;
  img[6] = 0;
  img[9] = 0;
  img[10] = 0;

  auto labels = RunComponents(img, w, h);

  int component_label = labels[5];
  EXPECT_GT(component_label, 0);
  EXPECT_EQ(labels[6], component_label);
  EXPECT_EQ(labels[9], component_label);
  EXPECT_EQ(labels[10], component_label);
  EXPECT_EQ(CountUniqueComponents(labels), 1);
}

TEST(dudchenko_o_connected_components_omp, u_shaped_component) {
  const int w = 5;
  const int h = 5;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);

  for (int y = 1; y <= 3; ++y) {
    img[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + 1] = 0;
    img[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + 3] = 0;
  }
  img[(3 * static_cast<std::size_t>(w)) + 2] = 0;

  auto labels = RunComponents(img, w, h);

  const int component_label = labels[(1 * static_cast<std::size_t>(w)) + 1];
  EXPECT_GT(component_label, 0);

  const std::vector<std::pair<int, int>> component_points = {
    {2, 1}, {3, 1}, {1, 3}, {2, 3}, {3, 2}
  };

  for (const auto& point : component_points) {
    const int x = point.first;
    const int y = point.second;
    EXPECT_EQ(labels[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + x], component_label);
  }

  EXPECT_EQ(CountUniqueComponents(labels), 1);
}