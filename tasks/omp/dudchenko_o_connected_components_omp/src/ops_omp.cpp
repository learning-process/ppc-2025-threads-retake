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
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::LabelComponents() {
  size_t total_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  output_.resize(total_pixels, 0);
  
  int num_threads = omp_get_max_threads();
  std::vector<std::vector<int>> local_labels(num_threads, std::vector<int>(total_pixels, 0));
  std::vector<ParentStructure> local_parents(num_threads);
  
  InitializeLocalParents(local_parents, total_pixels);
  
  std::vector<BlockRange> blocks(num_threads);
  CalculateBlockBoundaries(blocks, num_threads);

  ProcessBlocksInParallel(local_labels, local_parents, blocks, total_pixels, num_threads);
  MergeBlocks(local_labels, local_parents, blocks, total_pixels);
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::InitializeLocalParents(
    std::vector<ParentStructure>& local_parents, size_t total_pixels) {
  for (size_t i = 0; i < local_parents.size(); ++i) {
    local_parents[i].parents.resize(total_pixels + 1, 0);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::CalculateBlockBoundaries(
    std::vector<BlockRange>& blocks, int num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    int block_height = height_ / num_threads;
    blocks[i].start_y = i * block_height;
    blocks[i].end_y = (i == num_threads - 1) ? height_ : blocks[i].start_y + block_height;
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessBlocksInParallel(
    std::vector<std::vector<int>>& local_labels,
    std::vector<ParentStructure>& local_parents,
    const std::vector<BlockRange>& blocks,
    size_t total_pixels, int num_threads) {
#pragma omp parallel for
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    int base_label = (thread_id * (static_cast<int>(total_pixels) / num_threads)) + 1;
    ProcessBlock(blocks[thread_id], local_labels[thread_id], local_parents[thread_id], base_label);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessPixel(int x, int y, std::vector<int>& labels,
                                                                        ParentStructure& parent_structure,
                                                                        int& local_next_label) {
  int index = (y * width_) + x;

  if (input_[index] != 0) {
    labels[index] = 0;
    return;
  }

  int left_label = (x > 0) ? labels[index - 1] : 0;
  int top_label = (y > 0) ? labels[index - width_] : 0;

  if (left_label == 0 && top_label == 0) {
    labels[index] = local_next_label;
    parent_structure.parents[local_next_label] = local_next_label;
    local_next_label++;
  } else if (left_label != 0 && top_label == 0) {
    labels[index] = left_label;
  } else if (left_label == 0 && top_label != 0) {
    labels[index] = top_label;
  } else {
    ProcessConnectedNeighbors(left_label, top_label, labels, parent_structure, index);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessConnectedNeighbors(
    int left_label, int top_label, std::vector<int>& labels,
    ParentStructure& parent_structure, int index) {
  int root_left = FindRoot(parent_structure, left_label);
  int root_top = FindRoot(parent_structure, top_label);
  int min_root = (root_left < root_top) ? root_left : root_top;
  labels[index] = min_root;

  if (root_left != root_top) {
    UnionSets(parent_structure, root_left, root_top);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ProcessBlock(const BlockRange& block,
                                                                        std::vector<int>& labels,
                                                                        ParentStructure& parent_structure,
                                                                        int base_label) {
  int local_next_label = base_label;

  for (int y = block.start_y; y < block.end_y; ++y) {
    for (int x = 0; x < width_; ++x) {
      ProcessPixel(x, y, labels, parent_structure, local_next_label);
    }
  }

  NormalizeBlockLabels(block, labels, parent_structure);
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::NormalizeBlockLabels(
    const BlockRange& block, std::vector<int>& labels, ParentStructure& parent_structure) {
  for (int y = block.start_y; y < block.end_y; ++y) {
    for (int x = 0; x < width_; ++x) {
      int index = (y * width_) + x;
      if (labels[index] != 0) {
        labels[index] = FindRoot(parent_structure, labels[index]);
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::MergeBlocks(
    const std::vector<std::vector<int>>& local_labels,
    const std::vector<ParentStructure>& local_parents,
    const std::vector<BlockRange>& blocks, size_t total_pixels) {
  
  ParentStructure global_parent;
  global_parent.parents.resize(total_pixels + 1, 0);
  
  InitializeGlobalParent(global_parent);
  CopyLocalLabelsToOutput(local_labels, blocks);
  ResolveBlockBoundaries(global_parent, blocks);
  ResolveHorizontalConnections(global_parent);
  FinalNormalization(global_parent);
  RemapLabels();
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::InitializeGlobalParent(ParentStructure& global_parent) {
  for (size_t i = 1; i < global_parent.parents.size(); ++i) {
    global_parent.parents[i] = i;
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::CopyLocalLabelsToOutput(
    const std::vector<std::vector<int>>& local_labels, const std::vector<BlockRange>& blocks) {
  int num_threads = blocks.size();
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    const BlockRange& block = blocks[thread_id];
    for (int y = block.start_y; y < block.end_y; ++y) {
      for (int x = 0; x < width_; ++x) {
        int index = (y * width_) + x;
        output_[index] = local_labels[thread_id][index];
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveBlockBoundaries(
    ParentStructure& global_parent, const std::vector<BlockRange>& blocks) {
  int num_threads = blocks.size();
  for (int block_idx = 1; block_idx < num_threads; ++block_idx) {
    ResolveSingleBoundary(global_parent, blocks, block_idx);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveSingleBoundary(
    ParentStructure& global_parent, const std::vector<BlockRange>& blocks, int block_idx) {
  int boundary_y = blocks[block_idx].start_y;
  if (boundary_y > 0) {
    for (int x = 0; x < width_; ++x) {
      int top_index = ((boundary_y - 1) * width_) + x;
      int current_index = (boundary_y * width_) + x;
      
      if (output_[top_index] != 0 && output_[current_index] != 0) {
        int root_top = FindRoot(global_parent, output_[top_index]);
        int root_current = FindRoot(global_parent, output_[current_index]);
        
        if (root_top != root_current) {
          UnionSets(global_parent, root_top, root_current);
        }
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveHorizontalConnections(ParentStructure& global_parent) {
  for (int y = 0; y < height_; ++y) {
    ResolveHorizontalConnectionsInRow(global_parent, y);
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ResolveHorizontalConnectionsInRow(
    ParentStructure& global_parent, int y) {
  for (int x = 1; x < width_; ++x) {
    int left_index = (y * width_) + (x - 1);
    int current_index = (y * width_) + x;
    
    if (output_[left_index] != 0 && output_[current_index] != 0) {
      int root_left = FindRoot(global_parent, output_[left_index]);
      int root_current = FindRoot(global_parent, output_[current_index]);
      
      if (root_left != root_current) {
        UnionSets(global_parent, root_left, root_current);
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::FinalNormalization(ParentStructure& global_parent) {
  for (size_t i = 0; i < output_.size(); ++i) {
    if (output_[i] != 0) {
      output_[i] = FindRoot(global_parent, output_[i]);
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::RemapLabels() {
  std::vector<int> label_map(output_.size() + 1, 0);
  int next_label = 1;
  
  BuildLabelMap(label_map, next_label);
  ApplyLabelMap(label_map);
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::BuildLabelMap(std::vector<int>& label_map, int& next_label) {
  for (size_t i = 0; i < output_.size(); ++i) {
    if (output_[i] != 0) {
      if (label_map[output_[i]] == 0) {
        label_map[output_[i]] = next_label++;
      }
    }
  }
}

void dudchenko_o_connected_components_omp::TestTaskOpenMP::ApplyLabelMap(const std::vector<int>& label_map) {
  for (size_t i = 0; i < output_.size(); ++i) {
    if (output_[i] != 0) {
      output_[i] = label_map[output_[i]];
    }
  }
}

int dudchenko_o_connected_components_omp::TestTaskOpenMP::FindRoot(const ParentStructure& parent, int x) const {
  if (x <= 0 || static_cast<size_t>(x) >= parent.parents.size()) {
    return x;
  }
  int root = x;
  while (parent.parents[root] != root) {
    root = parent.parents[root];
  }
  
  // Path compression
  int temp = x;
  while (temp != root) {
    int next = parent.parents[temp];
    const_cast<ParentStructure&>(parent).parents[temp] = root;
    temp = next;
  }
  
  return root;
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