#include "omp/dudchenko_o_connected_components_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool dudchenko_o_connected_components_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = in_ptr[0];
  height_ = in_ptr[1];

  size_t pixel_count = width_ * height_;
  input_ = std::vector<int>(in_ptr + 2, in_ptr + 2 + pixel_count);
  output_ = std::vector<int>(pixel_count, 0);

  return true;
}

bool dudchenko_o_connected_components_omp::TestTaskOpenMP::ValidationImpl() {
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

bool dudchenko_o_connected_components_omp::TestTaskOpenMP::RunImpl() {
  if (input_.empty()) {
    return false;
  }

  LabelComponents();
  return true;
}

bool dudchenko_o_connected_components_omp::TestTaskOpenMP::PostProcessingImpl() {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(output_.size()); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::LabelComponents() {
  size_t total_pixels = width_ * height_;
  ComponentLabels labels;
  labels.labels.resize(total_pixels, 0);
  ParentStructure parent;
  parent.parents.resize(total_pixels + 1, 0);

  parent.parents[0] = 1;

  FirstPass(labels, parent);
  SecondPass(labels, parent);
  output_ = labels.labels;
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessPixel(int x, int y, int& local_next_label,
                                                                        ComponentLabels& component_labels,
                                                                        ParentStructure& parent_structure) {
  int index = (y * width_) + x;

  if (input_[index] != 0) {
    component_labels.labels[index] = 0;
    return;
  }

  int left_label = (x > 0) ? component_labels.labels[index - 1] : 0;
  int top_label = (y > 0) ? component_labels.labels[index - width_] : 0;

  if (left_label == 0 && top_label == 0) {
    int new_label = 0;
#pragma omp atomic capture
    new_label = parent_structure.parents[0]++;

    if (new_label >= static_cast<int>(parent_structure.parents.size())) {
#pragma omp critical
      {
        if (new_label >= static_cast<int>(parent_structure.parents.size())) {
          parent_structure.parents.resize(new_label + 100, 0);
        }
      }
    }
    parent_structure.parents[new_label] = new_label;
    component_labels.labels[index] = new_label;
    return;
  }

  if (left_label != 0 && top_label == 0) {
    component_labels.labels[index] = left_label;
    return;
  }
  if (left_label == 0 && top_label != 0) {
    component_labels.labels[index] = top_label;
    return;
  }

  int root_left = FindRoot(parent_structure, left_label);
  int root_top = FindRoot(parent_structure, top_label);

  if (root_left == root_top) {
    component_labels.labels[index] = root_left;
  } else {
    int min_root = std::min(root_left, root_top);
    component_labels.labels[index] = min_root;
    UnionSets(parent_structure, root_left, root_top);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::FirstPass(ComponentLabels& component_labels,
                                                                     ParentStructure& parent_structure) {
  int num_threads = omp_get_max_threads();
  int block_height = (height_ + num_threads - 1) / num_threads;

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_y = thread_id * block_height;
    int end_y = std::min(start_y + block_height, height_);
    int local_unused = 0;

    for (int y = start_y; y < end_y; ++y) {
      for (int x = 0; x < width_; ++x) {
        ProcessPixel(x, y, local_unused, component_labels, parent_structure);
      }
    }
  }

  ResolveBlockBoundaries(component_labels, parent_structure, block_height);
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveBlockBoundaries(ComponentLabels& component_labels,
                                                                                  ParentStructure& parent_structure,
                                                                                  int block_height) const {
  int num_blocks = (height_ + block_height - 1) / block_height;

  for (int block = 1; block < num_blocks; ++block) {
    int boundary_y = block * block_height;
    if (boundary_y >= height_) {
      continue;
    }

    for (int x = 0; x < width_; ++x) {
      int top_index = ((boundary_y - 1) * width_) + x;
      int current_index = (boundary_y * width_) + x;

      if (top_index < 0 || top_index >= static_cast<int>(component_labels.labels.size()) || current_index < 0 ||
          current_index >= static_cast<int>(component_labels.labels.size())) {
        continue;
      }

      int top_label = component_labels.labels[top_index];
      int current_label = component_labels.labels[current_index];

      if (top_label != 0 && current_label != 0) {
        int root_top = FindRoot(parent_structure, top_label);
        int root_current = FindRoot(parent_structure, current_label);

        if (root_top != root_current) {
          UnionSets(parent_structure, root_top, root_current);
        }
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::SecondPass(ComponentLabels& component_labels,
                                                                      ParentStructure& parent_structure) {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(component_labels.labels.size()); ++i) {
    if (component_labels.labels[i] != 0) {
      component_labels.labels[i] = FindRoot(parent_structure, component_labels.labels[i]);
    }
  }
}

int dudchenko_o_connected_components_omp::TestTaskOpenMP::FindRoot(ParentStructure& parent, int x) {
  if (x <= 0 || x >= static_cast<int>(parent.parents.size())) {
    return x;
  }

  int root = x;
  int iterations = 0;
  const int max_iterations = parent.parents.size();

  while (root > 0 && root < static_cast<int>(parent.parents.size()) && parent.parents[root] != root &&
         iterations < max_iterations) {
    root = parent.parents[root];
    iterations++;
  }

  int current = x;
  iterations = 0;
  while (current != root && current > 0 && current < static_cast<int>(parent.parents.size()) &&
         iterations < max_iterations) {
    int next = parent.parents[current];
    parent.parents[current] = root;
    current = next;
    iterations++;
  }

  return root;
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::UnionSets(ParentStructure& parent, int x, int y) {
  int root_x = FindRoot(parent, x);
  int root_y = FindRoot(parent, y);

  if (root_x == root_y) {
    return;
  }

  int min_root = 0;
  int max_root = 0;

  if (root_x < root_y) {
    min_root = root_x;
    max_root = root_y;
  } else {
    min_root = root_y;
    max_root = root_x;
  }

#pragma omp atomic write
  parent.parents[max_root] = min_root;
}