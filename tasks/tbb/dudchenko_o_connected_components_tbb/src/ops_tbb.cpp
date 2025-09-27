#include "tbb/dudchenko_o_connected_components_tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <unordered_map>
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

  int max_label = 0;
  for (int label : labels) {
    if (label > max_label) max_label = label;
  }
  if (max_label == 0) return;

  DisjointSetUnion dsu(max_label + 1);

  for (auto& eq : equivalences) {
    if (eq.first > 0 && eq.first <= max_label && eq.second > 0 && eq.second <= max_label) {
      dsu.unite(eq.first, eq.second);
    }
  }

  std::vector<int> label_map(max_label + 1, 0);
  int current_label = 1;
  for (int i = 1; i <= max_label; ++i) {
    int root = dsu.find(i);
    if (label_map[root] == 0) {
      label_map[root] = current_label++;
    }
  }

  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != 0) {
      labels[i] = label_map[dsu.find(labels[i])];
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

  int global_next_label = 1;
  std::mutex global_label_mutex;

  std::vector<std::pair<int, int>> all_equivalences;
  std::mutex eq_mutex;

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int>& range) {
    int local_next_label = 0;
    std::vector<std::pair<int, int>> local_equivalences;
    std::unordered_map<int, int> global_to_local;
    std::unordered_map<int, int> local_to_global;

    for (int y = range.begin(); y < range.end(); ++y) {
      for (int x = 0; x < width_; ++x) {
        size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);

        if (input_image_[idx] == 0) {
          output_labels_[idx] = 0;
          continue;
        }

        int left_label = (x > 0) ? output_labels_[idx - 1] : 0;
        int top_label = (y > 0) ? output_labels_[idx - static_cast<size_t>(width_)] : 0;

        int local_left = 0, local_top = 0;
        if (left_label != 0) {
          if (global_to_local.find(left_label) == global_to_local.end()) {
            local_next_label++;
            global_to_local[left_label] = local_next_label;
            local_to_global[local_next_label] = left_label;
          }
          local_left = global_to_local[left_label];
        }
        if (top_label != 0) {
          if (global_to_local.find(top_label) == global_to_local.end()) {
            local_next_label++;
            global_to_local[top_label] = local_next_label;
            local_to_global[local_next_label] = top_label;
          }
          local_top = global_to_local[top_label];
        }

        if (local_left == 0 && local_top == 0) {
          std::lock_guard<std::mutex> lock(global_label_mutex);
          int new_global_label = global_next_label++;
          output_labels_[idx] = new_global_label;
          global_to_local[new_global_label] = ++local_next_label;
          local_to_global[local_next_label] = new_global_label;
        } else if (local_left != 0 && local_top == 0) {
          output_labels_[idx] = local_to_global[local_left];
        } else if (local_left == 0 && local_top != 0) {
          output_labels_[idx] = local_to_global[local_top];
        } else {
          int min_global = std::min(local_to_global[local_left], local_to_global[local_top]);
          int max_global = std::max(local_to_global[local_left], local_to_global[local_top]);
          output_labels_[idx] = min_global;
          if (min_global != max_global) {
            local_equivalences.push_back({max_global, min_global});
          }
        }
      }
    }

    if (!local_equivalences.empty()) {
      std::lock_guard<std::mutex> lock(eq_mutex);
      all_equivalences.insert(all_equivalences.end(), local_equivalences.begin(), local_equivalences.end());
    }
  });

  ResolveEquivalences(output_labels_, all_equivalences);

  std::vector<int> unique_labels;
  for (int label : output_labels_) {
    if (label != 0 && std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
      unique_labels.push_back(label);
    }
  }
  std::sort(unique_labels.begin(), unique_labels.end());

  std::unordered_map<int, int> relabel_map;
  for (size_t i = 0; i < unique_labels.size(); ++i) {
    relabel_map[unique_labels[i]] = static_cast<int>(i + 1);
  }

  for (size_t i = 0; i < output_labels_.size(); ++i) {
    if (output_labels_[i] != 0) {
      output_labels_[i] = relabel_map[output_labels_[i]];
    }
  }

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