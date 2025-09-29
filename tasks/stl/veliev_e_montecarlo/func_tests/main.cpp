#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/veliev_e_montecarlo/include/ops_seq.hpp"
#include "stl/veliev_e_montecarlo/include/ops_stl.hpp"

constexpr double kESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_stl, test_lin_fun) {
  double res = 8;
  veliev_func_stl::Func f = veliev_func_stl::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_stl(1, res);
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

  veliev_e_stl_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_stl->inputs_count.emplace_back(in1.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_stl->inputs_count.emplace_back(1);

  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));
  task_data_stl->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_stl::VelievEMonteCarloStl task_stl(task_data_stl);
  ASSERT_EQ(task_stl.Validation(), true);
  task_stl.PreProcessing();
  task_stl.Run();
  task_stl.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_stl[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_stl, Test_sum_of_sin) {
  double res = 5.67369;
  veliev_func_stl::Func f = veliev_func_stl::FsinxPsiny;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {-1, 2};
  std::vector<double> out_stl(1, res);
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

  veliev_e_stl_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_stl->inputs_count.emplace_back(in1.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_stl->inputs_count.emplace_back(1);

  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));
  task_data_stl->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_stl::VelievEMonteCarloStl task_stl(task_data_stl);
  ASSERT_EQ(task_stl.Validation(), true);
  task_stl.PreProcessing();
  task_stl.Run();
  task_stl.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_stl[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_stl, test_sum_of_cos) {
  double res = 6.22943;
  veliev_func_stl::Func f = veliev_func_stl::FcosxPcosy;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_stl(1, res);
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

  veliev_e_stl_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_stl->inputs_count.emplace_back(in1.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_stl->inputs_count.emplace_back(1);

  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));
  task_data_stl->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_stl::VelievEMonteCarloStl task_stl(task_data_stl);
  ASSERT_EQ(task_stl.Validation(), true);
  task_stl.PreProcessing();
  task_stl.Run();
  task_stl.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_stl[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_stl, test_x_mult_y) {
  double res = 2.25;
  veliev_func_stl::Func f = veliev_func_stl::Fxy;

  std::vector<double> in1 = {0, 1};
  std::vector<double> in2 = {0, 3};
  std::vector<double> out_stl(1, res);
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

  veliev_e_stl_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_stl->inputs_count.emplace_back(in1.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_stl->inputs_count.emplace_back(1);

  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));
  task_data_stl->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_stl::VelievEMonteCarloStl task_stl(task_data_stl);
  ASSERT_EQ(task_stl.Validation(), true);
  task_stl.PreProcessing();
  task_stl.Run();
  task_stl.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_stl[0], kESTIMATE);
}

TEST(veliev_e_monte_carlo_stl, test_x_mult_y_mult_y) {
  double res = 1.5;
  veliev_func_stl::Func f = veliev_func_stl::Fxyy;

  std::vector<double> in1 = {0, 3};
  std::vector<double> in2 = {0, 1};
  std::vector<double> out_stl(1, res);
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

  veliev_e_stl_test::VelievEMonteCarloSeq task_seq(task_data_seq);
  ASSERT_EQ(task_seq.Validation(), true);
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_stl->inputs_count.emplace_back(in1.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_stl->inputs_count.emplace_back(in2.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_stl->inputs_count.emplace_back(1);

  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));
  task_data_stl->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_stl::VelievEMonteCarloStl task_stl(task_data_stl);
  ASSERT_EQ(task_stl.Validation(), true);
  task_stl.PreProcessing();
  task_stl.Run();
  task_stl.PostProcessing();

  ASSERT_NEAR(out_seq[0], out_stl[0], kESTIMATE);
}