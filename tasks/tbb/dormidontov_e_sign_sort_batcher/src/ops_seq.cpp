#include "tbb/dormidontov_e_sign_sort_batcher/include/ops_seq.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

namespace dormidontov_e_sign_sort_batcher_tbb {
namespace {

constexpr uint8_t kFullbyte = 255;
enum class ByteShift : size_t {};
uint8_t GetByte(double d_value, ByteShift shift) {
  auto* ptr = reinterpret_cast<uint64_t*>(&d_value);
  auto u = *ptr;
  if ((u >> 63) != 0) {
    u = ~u;
  } else {
    u ^= 1UL << 63;
  }
  return (u >> static_cast<size_t>(shift)) & kFullbyte;
}
}  // namespace

void tbbTask::Sort() {
  if (input_.empty()) {
    return;
  }

  const int radix = 256;
  std::vector<size_t> count(radix, 0);

  for (int shift = 0; shift < 64; shift += 8) {
    std::ranges::fill(begin(count), end(count), 0UL);
    for (size_t i = 0; i < input_size_; i++) {
      uint8_t key = GetByte(input_[i], ByteShift(shift));
      count[key]++;
    }
    for (int j = 1; j < radix; j++) {
      count[j] += count[j - 1];
    }
    for (int i = static_cast<int>(input_size_) - 1; i >= 0; i--) {
      uint8_t key = GetByte(input_[i], ByteShift(shift));
      tmp_[--count[key]] = input_[i];
    }
    input_.swap(tmp_);
  }
}

void tbbTask::Merge(int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;
  Merge(low, mid);
  Merge(mid, high);

  tbb::parallel_for(tbb::blocked_range<int>(low, mid), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      if (input_[i] > input_[i + mid - low]) {
        std::swap(input_[i], input_[i + mid - low]);
      }
    }
  });
}

bool tbbTask::PreProcessingImpl() {
  input_size_ = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size_);
  tmp_.resize(input_size_);
  return true;
}

bool tbbTask::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool tbbTask::RunImpl() {
  Sort();
  Merge(0, static_cast<int>(input_size_));
  return true;
}

bool tbbTask::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
}  // namespace dormidontov_e_sign_sort_batcher_tbb