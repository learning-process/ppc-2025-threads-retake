#include "omp/dudchenko_o_connected_components_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

bool dudchenko_o_connected_components_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = in_ptr[0];
  height_ = in_ptr[1];

  size_t pixel_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
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
  size_t expected_size = 2 + (static_cast<size_t>(width) * static_cast<size_t>(height));
  size_t output_expected_size = static_cast<size_t>(width) * static_cast<size_t>(height);

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
  size_t total_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  ComponentLabels labels;
  labels.labels.resize(total_pixels, 0);
  ParentStructure parent;
  parent.parents.resize(total_pixels + 1, 0);

  FirstPass(labels, parent);
  SecondPass(labels, parent);
  output_ = labels.labels;
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessPixel(int x, int y, ComponentLabels& component_labels,
                                                                        ParentStructure& parent_structure,
                                                                        int& local_next_label) {
  int index = y * width_ + x;

  if (input_[index] != 0) {
    component_labels.labels[index] = 0;
    return;
  }

  int left_label = (x > 0) ? component_labels.labels[index - 1] : 0;
  int top_label = (y > 0) ? component_labels.labels[index - width_] : 0;

  if (left_label == 0 && top_label == 0) {
    component_labels.labels[index] = local_next_label;
#pragma omp critical
    { parent_structure.parents[local_next_label] = local_next_label; }
    local_next_label++;
  } else if (left_label != 0 && top_label == 0) {
    component_labels.labels[index] = left_label;
  } else if (left_label == 0 && top_label != 0) {
    component_labels.labels[index] = top_label;
  } else {
    ProcessConnectedNeighbors(left_label, top_label, component_labels, parent_structure, index);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessConnectedNeighbors(int left_label, int top_label,
                                                                                     ComponentLabels& component_labels,
                                                                                     ParentStructure& parent_structure,
                                                                                     int index) {
  int root_left = FindRoot(parent_structure, left_label);
  int root_top = FindRoot(parent_structure, top_label);
  int min_root = (root_left < root_top) ? root_left : root_top;
  component_labels.labels[index] = min_root;

  if (root_left != root_top) {
#pragma omp critical
    { UnionSets(parent_structure, root_left, root_top); }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessBlock(int start_y, int end_y,
                                                                        ComponentLabels& component_labels,
                                                                        ParentStructure& parent_structure,
                                                                        int base_label) {
  int local_next_label = base_label;

  for (int y = start_y; y < end_y; ++y) {
    for (int x = 0; x < width_; ++x) {
      ProcessPixel(x, y, component_labels, parent_structure, local_next_label);
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveBlockBoundaries(ComponentLabels& component_labels,
                                                                                  ParentStructure& parent_structure) {
#pragma omp parallel for
  for (int y = 1; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int index = y * width_ + x;

      if (component_labels.labels[index] == 0) {
        continue;
      }

      int top_label = component_labels.labels[index - width_];

      if (top_label != 0 && component_labels.labels[index] != 0) {
        int root_current = FindRoot(parent_structure, component_labels.labels[index]);
        int root_top = FindRoot(parent_structure, top_label);

        if (root_current != root_top) {
#pragma omp critical
          { UnionSets(parent_structure, root_current, root_top); }
        }
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::FirstPass(ComponentLabels& component_labels,
                                                                     ParentStructure& parent_structure) {
  size_t total_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  int num_threads = omp_get_max_threads();

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int block_height = height_ / num_threads;
    int start_y = thread_id * block_height;
    int end_y = (thread_id == num_threads - 1) ? height_ : start_y + block_height;

    int base_label = (thread_id + 1) * (static_cast<int>(total_pixels) / num_threads) + 1;
    ProcessBlock(start_y, end_y, component_labels, parent_structure, base_label);
  }

  ResolveBlockBoundaries(component_labels, parent_structure);
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
  if (x <= 0 || static_cast<size_t>(x) >= parent.parents.size()) {
    return x;
  }
  if (parent.parents[x] != x) {
    parent.parents[x] = FindRoot(parent, parent.parents[x]);
  }
  return parent.parents[x];
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::UnionSets(ParentStructure& parent, int x, int y) {
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