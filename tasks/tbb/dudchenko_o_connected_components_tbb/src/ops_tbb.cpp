#include "tbb/dudchenko_o_connected_components_tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <mutex>

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
    dsu.unite(eq.first, eq.second);
  }
  
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != 0) {
      labels[i] = dsu.find(labels[i]);
    }
  }
}

static void RelabelComponents(std::vector<int>& labels, int& component_count) {
  int max_label = 0;
  for (int label : labels) {
    if (label > max_label) max_label = label;
  }
  
  if (max_label == 0) {
    component_count = 0;
    return;
  }
  
  std::vector<int> new_labels(static_cast<size_t>(max_label) + 1, 0);
  int current_label = 1;
  
  for (int label : labels) {
    if (label != 0 && new_labels[static_cast<size_t>(label)] == 0) {
      new_labels[static_cast<size_t>(label)] = current_label++;
    }
  }
  
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != 0) {
      labels[i] = new_labels[static_cast<size_t>(labels[i])];
    }
  }
  
  component_count = current_label - 1;
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
  
  std::vector<std::pair<int, int>> all_equivalences;
  std::mutex eq_mutex;
  
  tbb::parallel_for(tbb::blocked_range<int>(0, height_),
    [&](const tbb::blocked_range<int>& range) {
      std::vector<std::pair<int, int>> local_equivalences;
      
      for (int y = range.begin(); y < range.end(); ++y) {
        for (int x = 0; x < width_; ++x) {
          size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
          
          if (input_image_[idx] == 0) {
            output_labels_[idx] = 0;
            continue;
          }
          
          int left = (x > 0) ? output_labels_[idx - 1] : 0;
          int top = (y > 0) ? output_labels_[idx - static_cast<size_t>(width_)] : 0;
          
          if (left == 0 && top == 0) {
            output_labels_[idx] = (y * width_ + x) + 1;
          } else if (left != 0 && top == 0) {
            output_labels_[idx] = left;
          } else if (left == 0 && top != 0) {
            output_labels_[idx] = top;
          } else {
            int min_label = std::min(left, top);
            int max_label = std::max(left, top);
            output_labels_[idx] = min_label;
            if (min_label != max_label) {
              local_equivalences.push_back({max_label, min_label});
            }
          }
        }
      }
      
      std::lock_guard<std::mutex> lock(eq_mutex);
      all_equivalences.insert(all_equivalences.end(), 
                             local_equivalences.begin(), local_equivalences.end());
    });
  
  ResolveEquivalences(output_labels_, all_equivalences);
  RelabelComponents(output_labels_, components_count_);
  
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