#include "tbb/dudchenko_o_connected_components_tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>

using namespace dudchenko_o_connected_components_tbb;

class DisjointSetUnion {
 private:
  std::vector<int> parent;
  std::vector<int> rank;

 public:
  DisjointSetUnion(int n) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
    }
  }

  int find(int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  void unite(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y) return;

    if (rank[x] < rank[y]) {
      parent[x] = y;
    } else if (rank[x] > rank[y]) {
      parent[y] = x;
    } else {
      parent[y] = x;
      rank[x]++;
    }
  }
};

static void ResolveEquivalences(std::vector<int>& labels, std::vector<std::pair<int, int>>& equivalences) {
  if (equivalences.empty()) return;

  int max_label = *std::max_element(labels.begin(), labels.end());
  if (max_label == 0) return;

  DisjointSetUnion dsu(max_label + 1);

  for (auto& eq : equivalences) {
    if (eq.first > 0 && eq.first <= max_label && eq.second > 0 && eq.second <= max_label) {
      dsu.unite(eq.first, eq.second);
    }
  }

  std::vector<int> new_labels(max_label + 1, 0);
  int current_label = 1;
  for (int i = 1; i <= max_label; ++i) {
    int root = dsu.find(i);
    if (new_labels[root] == 0) {
      new_labels[root] = current_label++;
    }
  }

  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != 0) {
      labels[i] = new_labels[dsu.find(labels[i])];
    }
  }
}

namespace dudchenko_o_connected_components_tbb {

bool ConnectedComponentsTbb::ValidationImpl() {
  if (task_data == nullptr) return false;
  if (task_data->inputs.size() < 3 || task_data->inputs_count.size() < 3) return false;

  const unsigned int n = task_data->inputs_count[0];
  if (n > 0 && task_data->inputs[0] == nullptr) return false;

  if (task_data->inputs_count[1] != 1 || task_data->inputs_count[2] != 1) return false;
  if (task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) return false;

  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);
  if (w <= 0 || h <= 0) return false;
  if (static_cast<unsigned long long>(w) * static_cast<unsigned long long>(h) != n) return false;

  if (task_data->outputs.empty() || task_data->outputs_count.empty()) return false;
  const unsigned int cap = task_data->outputs_count[0];
  return (cap == 0) || (task_data->outputs[0] != nullptr);
}

bool ConnectedComponentsTbb::PreProcessingImpl() {
  const auto* img = reinterpret_cast<const uint8_t*>(task_data->inputs[0]);
  width_ = *reinterpret_cast<const int*>(task_data->inputs[1]);
  height_ = *reinterpret_cast<const int*>(task_data->inputs[2]);

  const size_t image_size = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  input_image_.assign(img, img + image_size);

  return true;
}

bool ConnectedComponentsTbb::RunImpl() {
  const size_t total_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  output_labels_.resize(total_pixels, 0);

  if (input_image_.empty()) {
    components_count_ = 0;
    return true;
  }

  int next_label = 1;
  std::vector<std::pair<int, int>> equivalences;

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);

      if (input_image_[idx] == 0) {
        output_labels_[idx] = 0;
        continue;
      }

      int left_label = (x > 0) ? output_labels_[idx - 1] : 0;
      int top_label = (y > 0) ? output_labels_[idx - static_cast<size_t>(width_)] : 0;

      if (left_label == 0 && top_label == 0) {
        output_labels_[idx] = next_label++;
      } else if (left_label != 0 && top_label == 0) {
        output_labels_[idx] = left_label;
      } else if (left_label == 0 && top_label != 0) {
        output_labels_[idx] = top_label;
      } else {
        int min_label = std::min(left_label, top_label);
        int max_label = std::max(left_label, top_label);
        output_labels_[idx] = min_label;
        if (min_label != max_label) {
          equivalences.push_back({max_label, min_label});
        }
      }
    }
  }

  ResolveEquivalences(output_labels_, equivalences);

  std::vector<int> unique_labels;
  for (int label : output_labels_) {
    if (label != 0 && std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
      unique_labels.push_back(label);
    }
  }
  std::sort(unique_labels.begin(), unique_labels.end());

  std::vector<int> label_map(*std::max_element(output_labels_.begin(), output_labels_.end()) + 1, 0);
  for (size_t i = 0; i < unique_labels.size(); ++i) {
    label_map[unique_labels[i]] = static_cast<int>(i + 1);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, output_labels_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); ++i) {
      if (output_labels_[i] != 0) {
        output_labels_[i] = label_map[output_labels_[i]];
      }
    }
  });

  components_count_ = static_cast<int>(unique_labels.size());
  return true;
}

bool ConnectedComponentsTbb::PostProcessingImpl() {
  auto* out = reinterpret_cast<int*>(task_data->outputs[0]);
  const unsigned int cap = task_data->outputs_count[0];

  if (out == nullptr || cap == 0) {
    task_data->outputs_count[0] = 0;
    return true;
  }

  const std::size_t n = std::min<std::size_t>(output_labels_.size(), cap);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = output_labels_[i];
  }

  task_data->outputs_count[0] = static_cast<unsigned int>(n);
  return true;
}

}  // namespace dudchenko_o_connected_components_tbb