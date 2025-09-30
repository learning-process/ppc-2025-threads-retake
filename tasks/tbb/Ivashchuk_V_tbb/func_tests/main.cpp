#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../include/ops_tbb.hpp"
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace {

static void CreateIdentityMatrices(std::vector<std::complex<double>> &in, size_t count) {
  for (size_t i = 0; i < count; i++) {
    in[(i * count) + i] = std::complex<double>(1.0, 0.0);
    in[(count * count) + (i * count) + i] = std::complex<double>(1.0, 0.0);
  }
}

static void VerifyDiagonal(const std::vector<std::complex<double>> &out, size_t count) {
  for (size_t i = 0; i < count; i++) {
    EXPECT_NEAR(out[(i * count) + i].real(), 1.0, 1e-10);
    EXPECT_NEAR(out[(i * count) + i].imag(), 0.0, 1e-10);
  }
}

static void VerifyOffDiagonal(const std::vector<std::complex<double>> &out, size_t count) {
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < count; j++) {
      if (i != j) {
        EXPECT_NEAR(out[(i * count) + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[(i * count) + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

}  // namespace

TEST(ivashchuk_v_tbb, test_sparse_matmul_5x5) {
  constexpr size_t kCount = 5;
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  CreateIdentityMatrices(in, kCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  ivashchuk_v_tbb::SparseMatrixComplexCRS task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  VerifyDiagonal(out, kCount);
  VerifyOffDiagonal(out, kCount);
}

TEST(ivashchuk_v_tbb, test_sparse_matmul_10x10_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("tbb/example/data/test.txt"));
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  const size_t count = std::stoi(line);
  std::vector<std::complex<double>> in(2 * count * count, 0.0);
  std::vector<std::complex<double>> out(count * count, 0.0);

  CreateIdentityMatrices(in, count);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  ivashchuk_v_tbb::SparseMatrixComplexCRS task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  VerifyDiagonal(out, count);
  VerifyOffDiagonal(out, count);
}