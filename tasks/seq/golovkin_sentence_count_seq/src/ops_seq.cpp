// Golovkin Maksim
#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

golovkin_sentence_count_seq::SentenceCountSequential::SentenceCountSequential(ppc::core::TaskDataPtr task_data)
    : Task(std::move(task_data)) {}

bool golovkin_sentence_count_seq::SentenceCountSequential::PreProcessingImpl() {
  text_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]));
  count_ = 0;
  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::ValidationImpl() {
  return !text_.empty() && task_data->outputs_count[0] == 1;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::RunImpl() {
  for (size_t i = 0; i < text_.size(); ++i) {
    if (text_[i] == '.' || text_[i] == '?' || text_[i] == '!') {
      ++count_;
      while (i + 1 < text_.size() && (text_[i + 1] == '.' || text_[i + 1] == '?' || text_[i + 1] == '!')) {
        ++i;
      }
    }
  }
  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = count_;
  return true;
}