#include "seq/anikin_m_lexic_check/include/ops_seq.hpp"

bool anikin_m_lexic_check_seq::LexicCheckSequential::ValidationImpl() {
  if ((task_data->inputs[0] == nullptr) || (task_data->inputs[1] == nullptr)) {
    return false;
  }
  if (task_data->outputs[0] == nullptr) {
    return false;
  }
  return true;
}

bool anikin_m_lexic_check_seq::LexicCheckSequential::PreProcessingImpl() {
  unsigned int input0_size = task_data->inputs_count[0];
  auto *in0_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input0_ = std::string(in0_ptr, in0_ptr + input0_size);

  unsigned int input1_size = task_data->inputs_count[1];
  auto *in1_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  input1_ = std::string(in1_ptr, in1_ptr + input1_size);

  ret = 0;

  return true;
}

bool anikin_m_lexic_check_seq::LexicCheckSequential::RunImpl() {
  if (input0_.size() < input1_.size()) {
    ret = -1;
  } else if (input0_.size() > input1_.size()) {
    ret = 1;
  } else {
    for (int i = 0; i < input0_.size(); i++) {
      if (input0_[i] == input1_[i]) {
        continue;
      }
      if (input0_[i] < input1_[i]) {
        ret = -1;
        break;
      } else {
        ret = 1;
        break;
      }
    }
  }
  return true;
}

bool anikin_m_lexic_check_seq::LexicCheckSequential::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = ret;
  return true;
}
