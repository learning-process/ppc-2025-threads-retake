// Golovkin
#include "tbb/golovkin_sentence_count_tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

#include <cctype>
#include <cstddef>
#include <functional>
#include <utility>

#include "core/task/include/task.hpp"

golovkin_sentence_count_tbb::SentenceCountParallel::SentenceCountParallel(ppc::core::TaskDataPtr task_data)
    : ppc::core::Task(std::move(task_data)) {}

bool golovkin_sentence_count_tbb::SentenceCountParallel::PreProcessingImpl() {
  text_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]));
  return true;
}

bool golovkin_sentence_count_tbb::SentenceCountParallel::ValidationImpl() { return task_data->outputs_count[0] == 1; }

bool golovkin_sentence_count_tbb::SentenceCountParallel::IsSentenceEnd(size_t i) const {
  if (text_[i] != '.') {
    return true;
  }

  const size_t n = text_.size();
  if (i > 0 && i + 1 < n) {
    const auto prev = static_cast<unsigned char>(text_[i - 1]);
    const auto next = static_cast<unsigned char>(text_[i + 1]);

    if ((std::isdigit(prev) != 0) && (std::isdigit(next) != 0)) {
      return false;
    }
    if ((std::isalpha(prev) != 0) && (std::isalpha(next) != 0)) {
      return false;
    }
    if (text_[i - 1] != ' ' && text_[i + 1] != ' ') {
      return false;
    }
  }
  return true;
}

bool golovkin_sentence_count_tbb::SentenceCountParallel::RunImpl() {
  const size_t n = text_.size();
  if (n == 0) {
    count_ = 0;
    return true;
  }

  count_ = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n), 0,
      [&](const tbb::blocked_range<size_t>& r, int local_count) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          const char current_char = text_[i];
          if (current_char != '.' && current_char != '?' && current_char != '!') {
            continue;
          }

          if (i + 1 < n) {
            const char next_char = text_[i + 1];
            if (next_char == '.' || next_char == '?' || next_char == '!') {
              continue;
            }
          }

          bool is_valid_end = true;
          if (current_char == '.') {
            is_valid_end = IsSentenceEnd(i);
          }

          if (is_valid_end) {
            local_count++;
          }
        }
        return local_count;
      },
      std::plus<>());

  return true;
}

bool golovkin_sentence_count_tbb::SentenceCountParallel::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = count_;
  return true;
}