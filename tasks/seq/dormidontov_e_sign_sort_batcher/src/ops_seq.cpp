#include "seq/dormidontov_e_sign_sort_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dormidontov_e_sign_sort_batcher_seq {
namespace {

constexpr uint8_t kFullbyte = 255;
enum class ByteShift : size_t {};
uint8_t GetByte(double d_value, ByteShift shift) {
  auto* ptr = reinterpret_cast<uint64_t*>(&d_value);
  uint64_t u = *ptr;
  if ((u >> 63) != 0) {
    u = ~u;
  } else {
    u ^= 1UL << 63;
  }
  return (u >> static_cast<size_t>(shift)) & kFullbyte;
}
}  // namespace

void SeqTask::Sort() {
  if (input_.empty()) {
    return;
  }

  const int radix = 256;
  std::vector<size_t> count(radix, 0);

  for (int shift = 0; shift < 64; shift += 8) {
    std::fill(begin(count), end(count), 0UL);
    for (size_t i = 0; i < input_size; i++) {
      uint8_t key = GetByte(input_[i], ByteShift(shift));
      count[key]++;
    }
    for (int j = 1; j < radix; j++) {
      count[j] += count[j - 1];
    }
    for (int i = static_cast<int>(input_size) - 1; i >= 0; i--) {
      uint8_t key = GetByte(input_[i], ByteShift(shift));
      tmp_[--count[key]] = input_[i];
    }
    input_.swap(tmp_);
  }
}

void SeqTask::Merge(int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;
  Merge(low, mid);
  Merge(mid, high);

  for (int i = low; i < mid; ++i) {
    if (input_[i] > input_[i + mid - low]) {
      std::swap(input_[i], input_[i + mid - low]);
    }
  }
}

bool SeqTask::PreProcessingImpl() {
  input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  tmp_.resize(input_size);
  return true;
}

bool SeqTask::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SeqTask::RunImpl() {
  Sort();
  Merge(0, static_cast<int>(input_size));
  return true;
}

bool SeqTask::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
}  // namespace dormidontov_e_sign_sort_batcher_seq