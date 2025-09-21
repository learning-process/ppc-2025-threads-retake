// Golovkin
#include "omp/golovkin_sentence_count_omp/include/ops_omp.hpp"

#include <omp.h>

#include <cctype>
#include <cstddef>
#include <utility>

#include "core/task/include/task.hpp"

golovkin_sentence_count_omp::SentenceCountParallel::SentenceCountParallel(ppc::core::TaskDataPtr task_data)
    : ppc::core::Task(std::move(task_data)), count_(0) {}

bool golovkin_sentence_count_omp::SentenceCountParallel::PreProcessingImpl() {
  text_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]));
  count_ = 0;
  return true;
}

bool golovkin_sentence_count_omp::SentenceCountParallel::ValidationImpl() { return task_data->outputs_count[0] == 1; }

bool golovkin_sentence_count_omp::SentenceCountParallel::IsSentenceEnd(size_t i) const {
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

bool golovkin_sentence_count_omp::SentenceCountParallel::RunImpl() {
  const size_t n = text_.size();
  if (n == 0) {
    return true;
  }

  count_ = 0;
  const int num_threads = omp_get_max_threads();
  const size_t chunk_size = n / num_threads;

#pragma omp parallel reduction(+ : count_)
  {
    const int tid = omp_get_thread_num();
    const size_t start = tid * chunk_size;
    const size_t end = (tid == num_threads - 1) ? n - 1 : start + chunk_size - 1;
    const size_t real_start = (start > 0) ? start - 1 : 0;
    const size_t real_end = (end < n - 1) ? end + 1 : n - 1;

    for (size_t i = real_start; i <= real_end; ++i) {
      if (text_[i] == '.' || text_[i] == '?' || text_[i] == '!') {
        if (i + 1 >= n || (text_[i + 1] != '.' && text_[i + 1] != '?' && text_[i + 1] != '!')) {
          if (i >= start && i <= end) {
            if (text_[i] != '.') {
              count_++;
            } else {
              if (IsSentenceEnd(i)) {
                count_++;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool golovkin_sentence_count_omp::SentenceCountParallel::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = count_;
  return true;
}