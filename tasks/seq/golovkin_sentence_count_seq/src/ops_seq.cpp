// Golovkin
#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <utility>

#include "core/task/include/task.hpp"

golovkin_sentence_count_seq::SentenceCountSequential::SentenceCountSequential(ppc::core::TaskDataPtr task_data)
    : ppc::core::Task(std::move(task_data)) {}

bool golovkin_sentence_count_seq::SentenceCountSequential::PreProcessingImpl() {
  text_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]));
  count_ = 0;
  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::ValidationImpl() { return task_data->outputs_count[0] == 1; }

namespace {
bool IsSentenceEnd(const std::string& text, size_t i, size_t n) {
  if (text[i] != '.') {
    return true;
  }

  if (i > 0 && i + 1 < n) {
    const auto prev = static_cast<unsigned char>(text[i - 1]);
    const auto next = static_cast<unsigned char>(text[i + 1]);

    if ((std::isdigit(prev) != 0) && (std::isdigit(next) != 0)) {
      return false;
    }
    if ((std::isalpha(prev) != 0) && (std::isalpha(next) != 0)) {
      return false;
    }
    if (text[i - 1] != ' ' && text[i + 1] != ' ') {
      return false;
    }
  }

  return true;
}
}  // namespace

bool golovkin_sentence_count_seq::SentenceCountSequential::RunImpl() {
  size_t i = 0;
  const size_t n = text_.size();

  while (i < n) {
    while (i < n && text_[i] != '.' && text_[i] != '?' && text_[i] != '!') {
      ++i;
    }

    if (i >= n) {
      break;
    }

    const bool is_sentence_end = IsSentenceEnd(text_, i, n);

    if (is_sentence_end) {
      ++count_;
      while (i < n && (text_[i] == '.' || text_[i] == '?' || text_[i] == '!')) {
        ++i;
      }
    } else {
      ++i;
    }
  }

  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = count_;
  return true;
}

bool golovkin_sentence_count_seq::SentenceCountSequential::IsAbbreviation(size_t i, size_t n) const {
  if (text_[i] != '.') {
    return false;
  }

  if (i > 0 && i + 1 < n) {
    const auto prev = static_cast<unsigned char>(text_[i - 1]);
    const auto next = static_cast<unsigned char>(text_[i + 1]);

    if ((std::isdigit(prev) != 0) && (std::isdigit(next) != 0)) {
      return true;
    }
    if ((std::isalpha(prev) != 0) && (std::isalpha(next) != 0)) {
      return true;
    }
    if (text_[i - 1] != ' ' && text_[i + 1] != ' ') {
      return true;
    }
  }

  return false;
}