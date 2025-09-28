#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool dudchenko_o_connected_components::TestTaskSequential::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = in_ptr[0];
  height_ = in_ptr[1];

  size_t pixel_count = width_ * height_;
  input_ = std::vector<int>(in_ptr + 2, in_ptr + 2 + pixel_count);
  output_ = std::vector<int>(pixel_count, 0);

  return true;
}

bool dudchenko_o_connected_components::TestTaskSequential::ValidationImpl() {
  if (!task_data || (task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  int width = in_ptr[0];
  int height = in_ptr[1];
  size_t expected_size = 2 + (static_cast<size_t>(width) * height);
  size_t output_expected_size = static_cast<size_t>(width) * height;

  return task_data->inputs_count[0] >= expected_size && task_data->outputs_count[0] >= output_expected_size;
}

bool dudchenko_o_connected_components::TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    return false;
  }

  LabelComponents();
  return true;
}

bool dudchenko_o_connected_components::TestTaskSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }
  return true;
}

void dudchenko_o_connected_components::TestTaskSequential::LabelComponents() {
  size_t total_pixels = width_ * height_;
  ComponentLabels labels;
  labels.labels.resize(total_pixels, 0);
  ParentStructure parent;
  parent.parents.resize(total_pixels + 1, 0);

  FirstPass(labels, parent);
  SecondPass(labels, parent);
  output_ = labels.labels;
}

void dudchenko_o_connected_components::TestTaskSequential::FirstPass(ComponentLabels& component_labels,
                                                                     ParentStructure& parent_structure) {
  int next_label = 1;

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int index = (y * width_) + x;

      if (input_[index] == 0) {
        component_labels.labels[index] = 0;
        continue;
      }

      int left_label = (x > 0) ? component_labels.labels[index - 1] : 0;
      int top_label = (y > 0) ? component_labels.labels[index - width_] : 0;

      if (left_label == 0 && top_label == 0) {
        component_labels.labels[index] = next_label;
        parent_structure.parents[next_label] = next_label;
        next_label++;
      } else if (left_label != 0 && top_label == 0) {
        component_labels.labels[index] = left_label;
      } else if (left_label == 0 && top_label != 0) {
        component_labels.labels[index] = top_label;
      } else {
        int root_left = FindRoot(parent_structure, left_label);
        int root_top = FindRoot(parent_structure, top_label);
        int min_root = std::min(root_left, root_top);
        component_labels.labels[index] = min_root;

        if (root_left != root_top) {
          UnionSets(parent_structure, root_left, root_top);
        }
      }
    }
  }
}

void dudchenko_o_connected_components::TestTaskSequential::SecondPass(ComponentLabels& component_labels,
                                                                      const ParentStructure& parent_structure) {
  for (size_t i = 0; i < component_labels.labels.size(); ++i) {
    if (component_labels.labels[i] != 0) {
      component_labels.labels[i] = FindRoot(const_cast<ParentStructure&>(parent_structure), component_labels.labels[i]);
    }
  }
}

int dudchenko_o_connected_components::TestTaskSequential::FindRoot(ParentStructure& parent, int x) {
  if (parent.parents[x] != x) {
    parent.parents[x] = FindRoot(parent, parent.parents[x]);
  }
  return parent.parents[x];
}

void dudchenko_o_connected_components::TestTaskSequential::UnionSets(ParentStructure& parent, int x, int y) {
  int root_x = FindRoot(parent, x);
  int root_y = FindRoot(parent, y);

  if (root_x != root_y) {
    if (root_x < root_y) {
      parent.parents[root_y] = root_x;
    } else {
      parent.parents[root_x] = root_y;
    }
  }
}