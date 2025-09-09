#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/khokhlov_a_shell_seq/include/ops_seq.hpp"

namespace khokhlov_a_shell_seq{
bool checkSorted(std::vector<int> input) { return std::is_sorted(input.begin(), input.end()); }

std::vector<int> generate_random_vector(int size, int min, int max) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{min, max};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  generate(begin(vec), end(vec), gen);

  return vec;
}
}

void runTestRandom(int count) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::generate_random_vector(count, 1, 100);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.ValidationImpl(), true);
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::checkSorted(out));
}

TEST(khokhlov_a_shell_seq, Shell_Validation_Fail) {
  const int count = 10;

  // Create data
  std::vector<int> in1 = khokhlov_a_shell_seq::generate_random_vector(count, 1, 100);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.ValidationImpl(), false);
}

TEST(khokhlov_a_shell_seq, Shell_Random_10) {
  const int count = 10;

  runTestRandom(count);
}

TEST(khokhlov_a_shell_seq, Shell_Random_20) {
  const int count = 20;

  runTestRandom(count);
}

TEST(khokhlov_a_shell_seq, Shell_Random_50) {
  const int count = 50;

  runTestRandom(count);
}

TEST(khokhlov_a_shell_seq, Shell_Random_70) {
  const int count = 70;

  runTestRandom(count);
}

TEST(khokhlov_a_shell_seq, Shell_Random_100) {
  const int count = 100;
  runTestRandom(count);
}