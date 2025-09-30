#include "stl/ersoz_b_hoare_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace ersoz_b_hoare_sort_simple_merge_stl {

long long HoareSortSimpleMergeSTL::Partition(std::vector<int>& a, long long l, long long r) {
  int pivot = a[l + ((r - l) / 2)];
  long long i = l;
  long long j = r;
  while (i <= j) {
    while (a[i] < pivot) {
      ++i;
    }
    while (a[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(a[i], a[j]);
      ++i;
      --j;
    }
  }
  return i;
}

void HoareSortSimpleMergeSTL::QuickSortHoare(std::vector<int>& a, long long l, long long r) {
  if (l >= r) {
    return;
  }
  long long i = l;
  long long j = r;
  int pivot = a[l + ((r - l) / 2)];
  while (i <= j) {
    while (a[i] < pivot) {
      ++i;
    }
    while (a[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(a[i], a[j]);
      ++i;
      --j;
    }
  }
  if (l < j) {
    QuickSortHoare(a, l, j);
  }
  if (i < r) {
    QuickSortHoare(a, i, r);
  }
}

void HoareSortSimpleMergeSTL::MergeTwo(const std::vector<int>& src, Segment left, Segment right,
                                       std::vector<int>& dst) {
  if (dst.empty()) {
    return;
  }

  auto left_first = src.begin() + static_cast<long long>(left.begin);
  auto left_last = src.begin() + static_cast<long long>(left.end);
  auto right_first = src.begin() + static_cast<long long>(right.begin);
  auto right_last = src.begin() + static_cast<long long>(right.end);
  std::merge(left_first, left_last, right_first, right_last, dst.begin());
}

// Helper function to perform parallel sorting
void HoareSortSimpleMergeSTL::ParallelSort(std::vector<int>& data, std::size_t n, int available_threads) {
  constexpr std::size_t kMinSegmentSize = 1024;
  std::size_t num_segments = std::min(static_cast<std::size_t>(available_threads), n / kMinSegmentSize);
  num_segments = std::max<std::size_t>(num_segments, 2);

  std::vector<std::thread> threads;
  std::vector<std::pair<std::size_t, std::size_t>> segments;

  // Create segments
  std::size_t segment_size = n / num_segments;
  for (std::size_t i = 0; i < num_segments; ++i) {
    std::size_t start = i * segment_size;
    std::size_t end = (i == num_segments - 1) ? n : (i + 1) * segment_size;
    if (start < end) {
      segments.emplace_back(start, end);
    }
  }

  // Sort each segment in parallel
  for (const auto& segment : segments) {
    if (segment.second - segment.first > 1) {
      threads.emplace_back([&, segment]() {
        QuickSortHoare(data, static_cast<long long>(segment.first), static_cast<long long>(segment.second) - 1);
      });
    }
  }

  // Wait for all sorting threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Perform parallel merge
  PerformParallelMerge(data, segments);
}

// Helper function to perform parallel merge
void HoareSortSimpleMergeSTL::PerformParallelMerge(std::vector<int>& data,
                                                   std::vector<std::pair<std::size_t, std::size_t>>& segments) {
  std::size_t merge_step = 1;

  while (merge_step < segments.size()) {
    std::vector<std::thread> merge_threads;

    for (std::size_t i = 0; i < segments.size(); i += 2 * merge_step) {
      if (i + merge_step < segments.size()) {
        merge_threads.emplace_back([&, i, merge_step]() {
          std::size_t left_start = segments[i].first;
          std::size_t left_end = segments[i].second;
          std::size_t right_start = segments[i + merge_step].first;
          std::size_t right_end = segments[i + merge_step].second;

          std::vector<int> merged_result(right_end - left_start);
          MergeTwo(data, Segment{.begin = left_start, .end = left_end}, Segment{.begin = right_start, .end = right_end},
                   merged_result);

          // Copy back to data
          for (std::size_t j = 0; j < merged_result.size(); ++j) {
            data[left_start + j] = merged_result[j];
          }
        });
      }
    }

    // Wait for merge threads
    for (auto& thread : merge_threads) {
      thread.join();
    }

    // Update segments for next iteration
    for (std::size_t i = 0; i < segments.size(); i += 2 * merge_step) {
      if (i + merge_step < segments.size()) {
        segments[i].second = segments[i + merge_step].second;
      }
    }

    merge_step *= 2;
  }
}

// Helper function for two-thread approach
void HoareSortSimpleMergeSTL::TwoThreadSort(std::vector<int>& data, std::size_t left_size, std::size_t n) {
  std::thread left_thread([&]() { QuickSortHoare(data, 0, static_cast<long long>(left_size) - 1); });
  QuickSortHoare(data, static_cast<long long>(left_size), static_cast<long long>(n) - 1);
  left_thread.join();

  MergeTwo(data, Segment{.begin = 0, .end = left_size}, Segment{.begin = left_size, .end = n}, output_);
}

// Helper function for sequential sort
void HoareSortSimpleMergeSTL::SequentialSort(std::vector<int>& data, std::size_t left_size, std::size_t n) {
  if (left_size > 1) {
    QuickSortHoare(data, 0, static_cast<long long>(left_size) - 1);
  }
  if (n - left_size > 1) {
    QuickSortHoare(data, static_cast<long long>(left_size), static_cast<long long>(n) - 1);
  }

  MergeTwo(data, Segment{.begin = 0, .end = left_size}, Segment{.begin = left_size, .end = n}, output_);
}

bool HoareSortSimpleMergeSTL::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  auto n = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + n);
  output_.assign(n, 0);
  return true;
}

bool HoareSortSimpleMergeSTL::ValidationImpl() {
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool HoareSortSimpleMergeSTL::RunImpl() {
  std::size_t n = input_.size();
  if (n == 0) {
    output_.clear();
    return true;
  }
  if (n == 1) {
    output_[0] = input_[0];
    return true;
  }

  long long mid = Partition(input_, 0, static_cast<long long>(n - 1));
  // Clamp mid defensively
  if (mid < 0) {
    mid = 0;
  } else if (mid > static_cast<long long>(n)) {
    mid = static_cast<long long>(n);
  }

  const auto left_size = static_cast<std::size_t>(mid);
  const std::size_t right_size = n - left_size;

  int available_threads = 1;
  try {
    available_threads = ppc::util::GetPPCNumThreads();
  } catch (...) {
    available_threads = 1;  // Fallback: sequential
  }

  constexpr std::size_t kParallelThreshold = 2048;

  // Choose sorting strategy based on data size and available threads
  if (available_threads > 2 && n >= kParallelThreshold * 2) {
    std::vector<int> temp_data = input_;
    ParallelSort(temp_data, n, available_threads);
    output_ = temp_data;
  } else if (available_threads > 1 && left_size > 1 && right_size > 1 &&
             (left_size >= kParallelThreshold || right_size >= kParallelThreshold)) {
    TwoThreadSort(input_, left_size, n);
  } else {
    SequentialSort(input_, left_size, n);
  }

  return true;
}

bool HoareSortSimpleMergeSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (std::size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace ersoz_b_hoare_sort_simple_merge_stl
