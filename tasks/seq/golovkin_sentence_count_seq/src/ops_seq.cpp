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
  if (text[i] != '.') return true;

  if (i > 0 && i + 1 < n) {
    const unsigned char prev = static_cast<unsigned char>(text[i - 1]);
    const unsigned char next = static_cast<unsigned char>(text[i + 1]);

    if (std::isdigit(prev) && std::isdigit(next)) return false;
    if (std::isalpha(prev) && std::isalpha(next)) return false;
    if (text[i - 1] != ' ' && text[i + 1] != ' ') return false;
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

    if (i >= n) break;

    const bool is_sentence_end = IsSentenceEnd(text_, i, n);

    if (is_sentence_end) {
      ++count_;
      const char current_punct = text_[i];

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