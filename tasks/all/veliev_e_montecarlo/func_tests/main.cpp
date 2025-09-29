#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "all/veliev_e_montecarlo/include/ops_all.hpp"
#include "all/veliev_e_montecarlo/include/ops_seq.hpp"
#include "core/task/include/task.hpp"

constexpr double kESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_all, test_lin_fun) {
  boost::mpi::communicator world;

  double res = 8;
  veliev_func_all::Func f = veliev_func_all::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_all(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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

    veliev_e_all_test::VelievEMonteCarloSeq task_seq(task_data_seq);
    ASSERT_EQ(task_seq.Validation(), true);
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
  }

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_all->inputs_count.emplace_back(in1.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_all->inputs_count.emplace_back(in2.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_all->inputs_count.emplace_back(1);

  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  task_data_all->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_all::VelievEMonteCarloAll task_all(task_data_all);
  task_all.SetFunc(f);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out_seq[0], out_all[0], kESTIMATE);
  }
}

TEST(veliev_e_monte_carlo_all, Test_sum_of_sin) {
  boost::mpi::communicator world;

  double res = 5.67369;
  veliev_func_all::Func f = veliev_func_all::FsinxPsiny;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {-1, 2};
  std::vector<double> out_all(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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

    veliev_e_all_test::VelievEMonteCarloSeq task_seq(task_data_seq);
    ASSERT_EQ(task_seq.Validation(), true);
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
  }

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_all->inputs_count.emplace_back(in1.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_all->inputs_count.emplace_back(in2.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_all->inputs_count.emplace_back(1);

  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  task_data_all->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_all::VelievEMonteCarloAll task_all(task_data_all);
  task_all.SetFunc(f);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out_seq[0], out_all[0], kESTIMATE);
  }
}

TEST(veliev_e_monte_carlo_all, test_sum_of_cos) {
  boost::mpi::communicator world;

  double res = 6.22943;
  veliev_func_all::Func f = veliev_func_all::FcosxPcosy;

  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_all(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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

    veliev_e_all_test::VelievEMonteCarloSeq task_seq(task_data_seq);
    ASSERT_EQ(task_seq.Validation(), true);
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
  }

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_all->inputs_count.emplace_back(in1.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_all->inputs_count.emplace_back(in2.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_all->inputs_count.emplace_back(1);

  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  task_data_all->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_all::VelievEMonteCarloAll task_all(task_data_all);
  task_all.SetFunc(f);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out_seq[0], out_all[0], kESTIMATE);
  }
}

TEST(veliev_e_monte_carlo_all, test_x_mult_y) {
  boost::mpi::communicator world;

  double res = 2.25;
  veliev_func_all::Func f = veliev_func_all::Fxy;

  std::vector<double> in1 = {0, 1};
  std::vector<double> in2 = {0, 3};
  std::vector<double> out_all(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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

    veliev_e_all_test::VelievEMonteCarloSeq task_seq(task_data_seq);
    ASSERT_EQ(task_seq.Validation(), true);
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
  }

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_all->inputs_count.emplace_back(in1.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_all->inputs_count.emplace_back(in2.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_all->inputs_count.emplace_back(1);

  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  task_data_all->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_all::VelievEMonteCarloAll task_all(task_data_all);
  task_all.SetFunc(f);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out_seq[0], out_all[0], kESTIMATE);
  }
}

TEST(veliev_e_monte_carlo_all, test_x_mult_y_mult_y) {
  boost::mpi::communicator world;

  double res = 1.5;
  veliev_func_all::Func f = veliev_func_all::Fxyy;

  std::vector<double> in1 = {0, 3};
  std::vector<double> in2 = {0, 1};
  std::vector<double> out_all(1, res);
  std::vector<double> out_seq(1, res);

  int n = 100;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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

    veliev_e_all_test::VelievEMonteCarloSeq task_seq(task_data_seq);
    ASSERT_EQ(task_seq.Validation(), true);
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
  }

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_all->inputs_count.emplace_back(in1.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_all->inputs_count.emplace_back(in2.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_all->inputs_count.emplace_back(1);

  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  task_data_all->outputs_count.emplace_back(out_seq.size());

  veliev_e_monte_carlo_all::VelievEMonteCarloAll task_all(task_data_all);
  task_all.SetFunc(f);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out_seq[0], out_all[0], kESTIMATE);
  }
}