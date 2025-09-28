// Golovkin
#pragma once

#include <cstddef>
#include <string>

#include "core/task/include/task.hpp"

namespace golovkin_sentence_count_seq {

class SentenceCountSequential : public ppc::core::Task {
 public:
  explicit SentenceCountSequential(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  [[nodiscard]] bool IsAbbreviation(size_t i, size_t n) const;
  std::string text_;
  int count_{};
};

}  // namespace golovkin_sentence_count_seq