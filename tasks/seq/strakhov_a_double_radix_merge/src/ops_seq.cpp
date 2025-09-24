#include "seq/strakhov_a_double_radix_merge/include/ops_seq.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <vector>

bool strakhov_a_double_radix_merge_seq::DoubleRadixMergeSeq::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  unsigned int size = task_data->inputs_count[0];
  input_ = std::vector<double>(in_ptr, in_ptr + size);

  output_ = std::vector<double>(size, 0);

  return true;
}

bool strakhov_a_double_radix_merge_seq::DoubleRadixMergeSeq::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool strakhov_a_double_radix_merge_seq::DoubleRadixMergeSeq::RunImpl() {
  unsigned int size = task_data->inputs_count[0];
  int type_length = sizeof(double) * CHAR_BIT;
  std::vector<uint64_t> temp_vector = std::vector<uint64_t>(size);
  // float to uint
  uint64_t tempest;
  for (int i = 0; i < size; i++) {
    tempest = std::bit_cast<uint64_t>(input_[i]);
    if (tempest >> 63) {
      temp_vector[i] = tempest ^ ~0ULL;
    } else {
      temp_vector[i] = tempest ^ 0x8000000000000000ULL;
    }
  }
  std::vector<uint64_t> true_vector = std::vector<uint64_t>(size);
  std::vector<uint64_t> false_vector = std::vector<uint64_t>(size);
  uint64_t bit_mask;
  for (int i = 0; i < type_length; i++) {
    true_vector.clear();
    false_vector.clear();
    bit_mask = (1 << i);
    for (int j = 0; j < size; j++) {
      if ((temp_vector[j] & bit_mask) == bit_mask) {
        true_vector.push_back(temp_vector[j]);
      } else {
        false_vector.push_back(temp_vector[j]);
      }
    }
    temp_vector.clear();
    temp_vector.insert(temp_vector.end(), false_vector.begin(), false_vector.end());
    temp_vector.insert(temp_vector.end(), true_vector.begin(), true_vector.end());
  }
  // uint to float
  for (int i = 0; i < size; i++) {
    tempest = std::bit_cast<uint64_t>(input_[i]);
    if (temp_vector[i] >> 63) {
      output_[i] = std::bit_cast<double>(temp_vector[i] ^ 0x8000000000000000ULL);
    } else {
      output_[i] = std::bit_cast<double>(temp_vector[i] ^ ~0ULL);
    }
  }
  return true;
}

bool strakhov_a_double_radix_merge_seq::DoubleRadixMergeSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
