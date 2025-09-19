// Golovkins
#pragma once

#include <cstddef>
#include <string>

#include "core/task/include/task.hpp"

namespace golovkin_sentence_count_omp {

class SentenceCountParallel : public ppc::core::Task {
 public:
  explicit SentenceCountParallel(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  [[nodiscard]] bool IsSentenceEnd(size_t i) const;
  std::string text_;
  int count_{};
};
}  // namespace golovkin_sentence_count_omp
