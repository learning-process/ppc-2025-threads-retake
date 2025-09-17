#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

TEST(golovkin_sentence_count_seq, test_empty_string) {
  std::string text = "";
  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  golovkin_sentence_count_seq::SentenceCountSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_EQ(0, result);
}

TEST(golovkin_sentence_count_seq, test_multiple_sentences) {
  std::string text = "Hello world! How are you? I'm fine. The end.";
  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  golovkin_sentence_count_seq::SentenceCountSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_EQ(4, result);
}