#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

#include <cctype>
#include <string>
#include <utility>

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

    if (i >= n) break;

    bool is_sentence_end = true;

    if (text_[i] == '.') {
      is_sentence_end = !is_dot_part_of_other_construct(i, n);
    }

    if (is_sentence_end) {
      count_++;
      skip_consecutive_punctuation(i, n);
    } else {
      i++;
    }
  }

  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::is_dot_part_of_other_construct(size_t i, size_t n) {
  if (i > 0 && i + 1 < n) {
    bool prev_is_digit = std::isdigit(text_[i - 1]) != 0;
    bool next_is_digit = std::isdigit(text_[i + 1]) != 0;
    if (prev_is_digit && next_is_digit) return true;
  }

  if (i > 0 && i + 1 < n) {
    bool prev_is_alpha = std::isalpha(text_[i - 1]) != 0;
    bool next_is_alpha = std::isalpha(text_[i + 1]) != 0;
    if (prev_is_alpha && next_is_alpha) return true;
  }

  if (i > 0 && i + 1 < n) {
    bool prev_not_space = text_[i - 1] != ' ';
    bool next_not_space = text_[i + 1] != ' ';
    if (prev_not_space && next_not_space) return true;
  }

  return false;
}

void golovkin_sentence_count_seq::SentenceCountSequential::skip_consecutive_punctuation(size_t& i, size_t n) {
  char current_punct = text_[i];
  while (i < n && (text_[i] == current_punct || text_[i] == '.' || text_[i] == '?' || text_[i] == '!')) {
    i++;
  }
}

bool golovkin_sentence_count_seq::SentenceCountSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = count_;
  return true;
}