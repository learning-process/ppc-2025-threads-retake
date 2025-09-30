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

  if (available_threads > 2 && n >= kParallelThreshold * 2) {
    std::size_t num_segments = std::min(static_cast<std::size_t>(available_threads), n / 1024);
    if (num_segments < 2) num_segments = 2;

    std::vector<std::thread> threads;
    std::vector<std::pair<std::size_t, std::size_t>> segments;

    std::size_t segment_size = n / num_segments;
    for (std::size_t i = 0; i < num_segments; ++i) {
      std::size_t start = i * segment_size;
      std::size_t end = (i == num_segments - 1) ? n : (i + 1) * segment_size;
      if (start < end) {
        segments.emplace_back(start, end);
      }
    }

    for (const auto& segment : segments) {
      if (segment.second - segment.first > 1) {
        threads.emplace_back([&, segment]() {
          QuickSortHoare(input_, static_cast<long long>(segment.first), static_cast<long long>(segment.second) - 1);
        });
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
    threads.clear();

    std::vector<int> temp_result = input_;
    std::size_t merge_step = 1;

    while (merge_step < segments.size()) {
      for (std::size_t i = 0; i < segments.size(); i += 2 * merge_step) {
        if (i + merge_step < segments.size()) {
          std::size_t left_start = segments[i].first;
          std::size_t left_end = segments[i].second;
          std::size_t right_start = segments[i + merge_step].first;
          std::size_t right_end = segments[i + merge_step].second;

          threads.emplace_back([&, left_start, left_end, right_start, right_end]() {
            std::vector<int> merged_result(right_end - left_start);
            MergeTwo(temp_result, Segment{left_start, left_end}, Segment{right_start, right_end}, merged_result);

            for (std::size_t j = 0; j < merged_result.size(); ++j) {
              temp_result[left_start + j] = merged_result[j];
            }
          });
        }
      }

      for (auto& thread : threads) {
        thread.join();
      }
      threads.clear();

      for (std::size_t i = 0; i < segments.size(); i += 2 * merge_step) {
        if (i + merge_step < segments.size()) {
          segments[i].second = segments[i + merge_step].second;
        }
      }

      merge_step *= 2;
    }

    output_ = temp_result;
  } else if (available_threads > 1 && left_size > 1 && right_size > 1 &&
             (left_size >= kParallelThreshold || right_size >= kParallelThreshold)) {
    std::thread left_thread([&]() { QuickSortHoare(input_, 0, static_cast<long long>(left_size) - 1); });
    QuickSortHoare(input_, static_cast<long long>(left_size), static_cast<long long>(n) - 1);
    left_thread.join();

    MergeTwo(input_, Segment{.begin = 0, .end = left_size}, Segment{.begin = left_size, .end = n}, output_);
  } else {
    if (left_size > 1) {
      QuickSortHoare(input_, 0, static_cast<long long>(left_size) - 1);
    }
    if (right_size > 1) {
      QuickSortHoare(input_, static_cast<long long>(left_size), static_cast<long long>(n) - 1);
    }

    MergeTwo(input_, Segment{.begin = 0, .end = left_size}, Segment{.begin = left_size, .end = n}, output_);
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
