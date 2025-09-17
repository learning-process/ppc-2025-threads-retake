#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>

golovkin_sentence_count_seq::SentenceCountSequential::SentenceCountSequential(ppc::core::TaskDataPtr task_data)
    : Task(std::move(task_data)) {}

bool golovkin_sentence_count_seq::SentenceCountSequential::PreProcessingImpl() {
  text_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]));
  count_ = 0;
  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::ValidationImpl() { return task_data->outputs_count[0] == 1; }

bool golovkin_sentence_count_seq::SentenceCountSequential::RunImpl() {
  size_t i = 0;
  size_t n = text_.size();

  while (i < n) {
    while (i < n && text_[i] != '.' && text_[i] != '?' && text_[i] != '!') {
      i++;
    }

    if (i < n) {
      bool is_sentence_end = true;

      if (text_[i] == '.') {
        if (i > 0 && i + 1 < n && std::isdigit(text_[i - 1]) && std::isdigit(text_[i + 1])) {
          is_sentence_end = false;
        } else if (i > 0 && std::isalpha(text_[i - 1]) && i + 1 < n && std::isalpha(text_[i + 1])) {
          is_sentence_end = false;
        } else if (i > 0 && text_[i - 1] != ' ' && i + 1 < n && text_[i + 1] != ' ') {
          is_sentence_end = false;
        }
      }

      if (is_sentence_end) {
        count_++;

        char current_punct = text_[i];
        while (i < n && (text_[i] == current_punct || text_[i] == '.' || text_[i] == '?' || text_[i] == '!')) {
          i++;
        }
      } else {
        i++;
      }
    }
  }

  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = count_;
  return true;
}