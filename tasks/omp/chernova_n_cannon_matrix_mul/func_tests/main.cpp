#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/example/include/ops_omp.hpp"

bool compareMatrices(const std::vector<double> &matrix1, const std::vector<double> &matrix2, double tolerance = 1e-4) {
  if (matrix1.size() != matrix2.size()) {
    return false;
  }

  for (size_t i = 0; i < matrix1.size(); ++i) {
    if (std::fabs(matrix1[i] - matrix2[i]) > tolerance) {
      return false;
    }
  }

  return true;
}

std::vector<double> GetRandomMatrix(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-100.0, 100.0);

  std::vector<double> matrix(n * n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[(i * n) + j] = dis(gen);
    }
  }

  return matrix;
}

TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_2) {
  int n = 2;

  std::vector<double> matrix_a{1, 2, 3, 4};
  std::vector<double> matrix_b{6, 7, 8, 9};
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_4) {
  int n = 4;

  std::vector<double> matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_16) {
  int n = 16;

  std::vector<double> matrix_a = GetRandomMatrix(n);
  std::vector<double> matrix_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_100) {
  int n = 100;

  std::vector<double> matrix_a = GetRandomMatrix(n);
  std::vector<double> matrix_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_1000) {
  int n = 1000;

  std::vector<double> matrix_a = GetRandomMatrix(n);
  std::vector<double> matrix_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
TEST(chernova_n_cannon_matrix_mul_omp, test_matmul_1600) {
  int n = 1600;

  std::vector<double> matrix_a = GetRandomMatrix(n);
  std::vector<double> matrix_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(matrix_a, matrix_b, n);

  chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}