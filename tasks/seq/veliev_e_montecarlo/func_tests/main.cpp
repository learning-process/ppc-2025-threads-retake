#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/veliev_e_montecarlo/include/ops_seq.hpp"

constexpr double kESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_seq, test_lin_fun) {
  double res = 8;
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

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

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  veliev_e_monte_carlo_seq::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
}

TEST(veliev_e_monte_carlo_seq, Test_sum_of_sin) {
  double res = 5.67369;
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::FsinxPsiny;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {-1, 2};
  std::vector<double> out(1, res);

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

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  veliev_e_monte_carlo_seq::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
}

TEST(veliev_e_monte_carlo_seq, test_sum_of_cos) {
  double res = 6.22943;
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::FcosxPcosy;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

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

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  veliev_e_monte_carlo_seq::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
}

TEST(veliev_e_monte_carlo_seq, test_x_mult_y) {
  double res = 2.25;
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::Fxy;

  std::vector<double> in1 = {0, 1};
  std::vector<double> in2 = {0, 3};
  std::vector<double> out(1, res);

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

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  veliev_e_monte_carlo_seq::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
}

TEST(veliev_e_monte_carlo_seq, test_x_mult_y_mult_y) {
  double res = 1.5;
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::Fxyy;

  std::vector<double> in1 = {0, 3};
  std::vector<double> in2 = {0, 1};
  std::vector<double> out(1, res);

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

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  veliev_e_monte_carlo_seq::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
}