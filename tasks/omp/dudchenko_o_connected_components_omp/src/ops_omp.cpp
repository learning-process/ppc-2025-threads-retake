#include "omp/dudchenko_o_connected_components_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

using namespace dudchenko_o_connected_components_omp;

namespace {

constexpr uint8_t FOREGROUND = 0;

}  // namespace

bool ConnectedComponentsOmp::ValidationImpl() {
  if (task_data == nullptr) {
    return false;
  }

  if (task_data->inputs.size() < 3 || task_data->inputs_count.size() < 3) {
    return false;
  }

  const unsigned int image_size = task_data->inputs_count[0];
  if (image_size > 0 && task_data->inputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[1] != 1 || task_data->inputs_count[2] != 1) {
    return false;
  }
  if (task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) {
    return false;
  }

  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);

  if (w <= 0 || h <= 0) {
    return false;
  }
  if (static_cast<size_t>(w) * static_cast<size_t>(h) != image_size) {
    return false;
  }

  if (task_data->outputs.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  const unsigned int output_capacity = task_data->outputs_count[0];
  return (output_capacity == 0) || (task_data->outputs[0] != nullptr);
}

bool ConnectedComponentsOmp::PreProcessingImpl() {
  const auto* image_data = reinterpret_cast<const uint8_t*>(task_data->inputs[0]);
  width_ = *reinterpret_cast<const int*>(task_data->inputs[1]);
  height_ = *reinterpret_cast<const int*>(task_data->inputs[2]);

  const size_t image_size = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  input_image_.assign(image_data, image_data + image_size);
  output_labels_.resize(image_size, 0);

  return true;
}

bool ConnectedComponentsOmp::RunImpl() {
  const size_t image_size = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<int> labels(image_size, 0);

  int next_label = 1;
  std::vector<int> parent(1000, 0);

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);

      if (input_image_[idx] == FOREGROUND) {
        int left_label = 0;
        int top_label = 0;

        if (x > 0 && input_image_[idx - 1] == FOREGROUND) {
          left_label = labels[idx - 1];
        }
        if (y > 0 && input_image_[idx - width_] == FOREGROUND) {
          top_label = labels[idx - width_];
        }

        if (left_label == 0 && top_label == 0) {
          labels[idx] = next_label;
          if (static_cast<size_t>(next_label) >= parent.size()) {
            parent.resize(next_label * 2, 0);
          }
          parent[next_label] = next_label;
          next_label++;
        } else if (left_label != 0 && top_label == 0) {
          labels[idx] = left_label;
        } else if (left_label == 0 && top_label != 0) {
          labels[idx] = top_label;
        } else {
          int min_label = std::min(left_label, top_label);
          int max_label = std::max(left_label, top_label);
          labels[idx] = min_label;

          if (min_label != max_label) {
            int root_min = FindRoot(parent, min_label);
            int root_max = FindRoot(parent, max_label);

            if (root_min != root_max) {
                int new_root = std::min(root_min, root_max);
                int old_root = std::max(root_min, root_max);

#pragma omp atomic write
                parent[old_root] = new_root;

                if (min_label != root_min) {
#pragma omp atomic write
                    parent[min_label] = new_root;
                }
                if (max_label != root_max) {
#pragma omp atomic write
                    parent[max_label] = new_root;
                }
            }
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);

      if (labels[idx] > 0) {
        int label = labels[idx];
        while (parent[label] != label) {
          label = parent[label];
        }
        labels[idx] = label;
      }
    }
  }

  std::map<int, int> label_map;
  int current_label = 1;

  for (size_t i = 0; i < image_size; ++i) {
    if (labels[i] > 0) {
      label_map[labels[i]] = 0;
    }
  }

  for (auto& pair : label_map) {
    pair.second = current_label++;
  }

  components_count_ = current_label - 1;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(image_size); ++i) {
    if (labels[i] > 0) {
      output_labels_[i] = label_map[labels[i]];
    } else {
      output_labels_[i] = 0;
    }
  }

  return true;
}

bool ConnectedComponentsOmp::PostProcessingImpl() {
  auto* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
  const unsigned int output_capacity = task_data->outputs_count[0];
  const size_t data_size = output_labels_.size();

  if (output_data == nullptr || output_capacity == 0) {
    task_data->outputs_count[0] = 0;
    return true;
  }

  const size_t copy_size = std::min(static_cast<size_t>(output_capacity), data_size);
  for (size_t i = 0; i < copy_size; ++i) {
    output_data[i] = output_labels_[i];
  }

  task_data->outputs_count[0] = static_cast<unsigned int>(copy_size);

  return true;
}