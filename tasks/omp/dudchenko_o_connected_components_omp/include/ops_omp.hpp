#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_connected_components_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  struct ParentStructure {
    std::vector<int32_t> parents;
  };

  struct BlockRange {
    int start_y;
    int end_y;
  };

  std::vector<int> input_;
  std::vector<int> output_;
  int width_{};
  int height_{};

  void LabelComponents();

  static void InitializeLocalParents(std::vector<ParentStructure>& local_parents, size_t total_pixels);
  static void CalculateBlockBoundaries(std::vector<BlockRange>& blocks, int num_threads, int height);
  void ProcessBlocksInParallel(std::vector<std::vector<int>>& local_labels, std::vector<ParentStructure>& local_parents,
                               const std::vector<BlockRange>& blocks, size_t total_pixels, int num_threads);

  void ProcessPixel(int x, int y, std::vector<int>& labels, ParentStructure& parent_structure, int& local_next_label);
  void ProcessConnectedNeighbors(int left_label, int top_label, std::vector<int>& labels,
                                 ParentStructure& parent_structure, int index);
  void ProcessBlock(const BlockRange& block, std::vector<int>& labels, ParentStructure& parent_structure,
                    int base_label);
  void NormalizeBlockLabels(const BlockRange& block, std::vector<int>& labels, ParentStructure& parent_structure);

  void MergeBlocks(const std::vector<std::vector<int>>& local_labels, const std::vector<ParentStructure>& local_parents,
                   const std::vector<BlockRange>& blocks, size_t total_pixels);
  static void InitializeGlobalParent(ParentStructure& global_parent);
  void CopyLocalLabelsToOutput(const std::vector<std::vector<int>>& local_labels,
                               const std::vector<BlockRange>& blocks);
  void ResolveBlockBoundaries(ParentStructure& global_parent, const std::vector<BlockRange>& blocks);
  void ResolveSingleBoundary(ParentStructure& global_parent, const std::vector<BlockRange>& blocks, int block_idx);
  void ResolveHorizontalConnections(ParentStructure& global_parent);
  void ResolveHorizontalConnectionsInRow(ParentStructure& global_parent, int y);
  void FinalNormalization(ParentStructure& global_parent);

  void RemapLabels();
  void BuildLabelMap(std::vector<int>& label_map, int& next_label);
  void ApplyLabelMap(const std::vector<int>& label_map);

  [[nodiscard]] static int FindRoot(const ParentStructure& parent, int x);
  static void UnionSets(ParentStructure& parent, int x, int y);
};

}  // namespace dudchenko_o_connected_components_omp