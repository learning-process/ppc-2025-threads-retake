#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/anikin_m_lexic_check/include/ops_seq.hpp"

TEST(anikin_m_lexic_check_seq, test_pipeline_run) {
  // Create data
  std::string in0 = "dokg0wjgijwigjsdoigiwejg0wei0gjw0ejgi90sw90gse";
  std::string in1 = "fdpgjeigjiwegiweiogjiosdjgonijwegjweg";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_lexic_check_seq::LexicCheckSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true, true);
}

TEST(anikin_m_lexic_check_seq, test_task_run) {
  // Create data
  std::string in0 = "dokg0wjgijwigjsdoigiwejg0wei0gjw0ejgi90sw90gse";
  std::string in1 = "fdpgjeigjiwegiweiogjiosdjgonijwegjweg";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_lexic_check_seq::LexicCheckSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true, true);
}
