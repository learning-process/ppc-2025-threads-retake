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

TEST(golovkin_sentence_count_seq, test_no_sentences) {
  std::string text = "This is a text without sentence endings";
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

TEST(golovkin_sentence_count_seq, test_single_sentence_dot) {
  std::string text = "This is a sentence.";
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
  ASSERT_EQ(1, result);
}

TEST(golovkin_sentence_count_seq, test_single_sentence_question) {
  std::string text = "Is this a question?";
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
  ASSERT_EQ(1, result);
}

TEST(golovkin_sentence_count_seq, test_single_sentence_exclamation) {
  std::string text = "What a beautiful day!";
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
  ASSERT_EQ(1, result);
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

TEST(golovkin_sentence_count_seq, test_consecutive_punctuation) {
  std::string text = "Wow!!! Is this real?? Yes... It is.";
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

TEST(golovkin_sentence_count_seq, test_mixed_punctuation) {
  std::string text = "First sentence. Second! Third? Fourth.";
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

TEST(golovkin_sentence_count_seq, test_ellipsis) {
  std::string text = "This is an ellipsis... It should count as one sentence.";
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
  ASSERT_EQ(2, result);
}

TEST(golovkin_sentence_count_seq, test_whitespace_after_punctuation) {
  std::string text = "Sentence with space after period . And another one ! And question ?";
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
  ASSERT_EQ(3, result);
}

TEST(golovkin_sentence_count_seq, test_numbers_with_dots) {
  std::string text = "The value is 3.14. This is pi.";
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
  ASSERT_EQ(2, result);
}

TEST(golovkin_sentence_count_seq, test_email_and_url) {
  std::string text = "Contact me at email@example.com. Visit our website at https://example.com.";
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
  ASSERT_EQ(2, result);
}

TEST(golovkin_sentence_count_seq, test_long_text) {
  std::string text =
      "This is the first sentence. This is the second sentence! "
      "And this is the third sentence? Here comes the fourth sentence. "
      "Finally, the fifth sentence.";
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
  ASSERT_EQ(5, result);
}