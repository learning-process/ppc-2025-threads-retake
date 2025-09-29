#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/veliev_e_montecarlo/include/ops_seq.hpp"
#include "tbb/veliev_e_montecarlo/include/ops_tbb.hpp"

constexpr double kESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_tbb, test_lin_fun) {
  double res = 8;
  veliev_func_tbb::Func f = veliev_func_tbb::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_tbb(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_seq->outputs_count.emplace_back(out_seq.size());

  veliev_e_tbb_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_tbb->inputs_count.emplace_back(in1.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs_count.emplace_back(1);

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_tbb.data()));
  task_data_tbb->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_tbb[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_tbb, Test_sum_of_sin) {
  double res = 5.67369;
  veliev_func_tbb::Func f = veliev_func_tbb::FsinxPsiny;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {-1, 2};
  std::vector<double> out_tbb(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_seq->outputs_count.emplace_back(out_seq.size());

  veliev_e_tbb_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_tbb->inputs_count.emplace_back(in1.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs_count.emplace_back(1);

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_tbb.data()));
  task_data_tbb->outputs_count.emplace_back(out_seq.size());
  task_data_tbb->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_tbb[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_tbb, test_sum_of_cos) {
  double res = 6.22943;
  veliev_func_tbb::Func f = veliev_func_tbb::FcosxPcosy;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_tbb(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_seq->outputs_count.emplace_back(out_seq.size());

  veliev_e_tbb_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_tbb->inputs_count.emplace_back(in1.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs_count.emplace_back(1);

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_tbb.data()));
  task_data_tbb->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_tbb[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_tbb, test_x_mult_y) {
  double res = 2.25;
  veliev_func_tbb::Func f = veliev_func_tbb::Fxy;

  std::vector<double> in1 = {0, 1};
  std::vector<double> in2 = {0, 3};
  std::vector<double> out_tbb(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_seq->outputs_count.emplace_back(out_seq.size());

  veliev_e_tbb_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_tbb->inputs_count.emplace_back(in1.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs_count.emplace_back(1);

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_tbb.data()));
  task_data_tbb->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb task_omp(task_data_tbb);
  ASSERT_EQ(task_omp.Validation(), true);
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_tbb[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_tbb, test_x_mult_y_mult_y) {
  double res = 1.5;
  veliev_func_tbb::Func f = veliev_func_tbb::Fxyy;

  std::vector<double> in1 = {0, 3};
  std::vector<double> in2 = {0, 1};
  std::vector<double> out_tbb(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  task_data_seq->outputs_count.emplace_back(out_seq.size());

  veliev_e_tbb_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_tbb->inputs_count.emplace_back(in1.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_tbb->inputs_count.emplace_back(in2.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs_count.emplace_back(1);

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_tbb.data()));
  task_data_tbb->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb task_omp(task_data_tbb);
  ASSERT_EQ(task_omp.Validation(), true);
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_tbb[0], kESTIMATE);
}