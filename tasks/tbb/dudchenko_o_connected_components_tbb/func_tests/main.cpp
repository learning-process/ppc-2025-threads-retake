#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/dudchenko_o_connected_components_tbb/include/ops_tbb.hpp"

using dudchenko_o_connected_components_tbb::ConnectedComponentsTbb;

namespace {
std::vector<int> RunConnectedComponents(const std::vector<uint8_t>& img, int w, int h) {
  std::vector<int> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(img.data())));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto task = std::make_shared<ConnectedComponentsTbb>(td);

  if (!task->ValidationImpl()) return {};
  if (!task->PreProcessingImpl()) return {};
  if (!task->RunImpl()) return {};
  if (!task->PostProcessingImpl()) return {};

  const unsigned int n = td->outputs_count[0];
  return {out.begin(), out.begin() + n};
}

int CountUniqueComponents(const std::vector<int>& labels) {
  std::vector<int> unique_labels;
  for (int label : labels) {
    if (label != 0 && std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
      unique_labels.push_back(label);
    }
  }
  return static_cast<int>(unique_labels.size());
}
}  // namespace

TEST(dudchenko_o_connected_components_tbb, single_component) {
  const int w = 5, h = 5;
  std::vector<uint8_t> img(w * h, 0);
  
  for (int y = 1; y <= 3; ++y) {
    for (int x = 1; x <= 3; ++x) {
      img[y * w + x] = 1;
    }
  }

  auto labels = RunConnectedComponents(img, w, h);
  EXPECT_EQ(CountUniqueComponents(labels), 1);
}

TEST(dudchenko_o_connected_components_tbb, two_separate_components) {
  const int w = 6, h = 6;
  std::vector<uint8_t> img(w * h, 0);
  
  img[1 * w + 1] = 1; img[1 * w + 2] = 1;
  img[2 * w + 1] = 1; img[2 * w + 2] = 1;
  
  img[4 * w + 4] = 1; img[4 * w + 5] = 1;
  img[5 * w + 4] = 1; img[5 * w + 5] = 1;

  auto labels = RunConnectedComponents(img, w, h);
  EXPECT_EQ(CountUniqueComponents(labels), 2);
}

TEST(dudchenko_o_connected_components_tbb, empty_image) {
  const int w = 3, h = 3;
  std::vector<uint8_t> img(w * h, 0);

  auto labels = RunConnectedComponents(img, w, h);
  EXPECT_EQ(CountUniqueComponents(labels), 0);
}

TEST(dudchenko_o_connected_components_tbb, full_image) {
  const int w = 4, h = 4;
  std::vector<uint8_t> img(w * h, 1);

  auto labels = RunConnectedComponents(img, w, h);
  EXPECT_EQ(CountUniqueComponents(labels), 1);
}