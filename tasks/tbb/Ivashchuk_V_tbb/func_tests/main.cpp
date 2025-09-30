#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "include/ops_tbb.hpp"

TEST(ivashchuk_v_tbb, test_sparse_matmul_5x5) {
  constexpr size_t kCount = 5;

  // Create data - two 5x5 matrices with complex numbers
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  // Create identity matrices for testing
  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = std::complex<double>(1.0, 0.0);                      // First matrix (identity)
    in[(kCount * kCount) + (i * kCount) + i] = std::complex<double>(1.0, 0.0);  // Second matrix (identity)
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_tbb::SparseMatrixComplexCRS test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  // Check result (identity * identity = identity)
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[(i * kCount) + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[(i * kCount) + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(ivashchuk_v_tbb, test_sparse_matmul_10x10_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("tbb/example/data/test.txt"));
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  const size_t count = std::stoi(line);

  // Create data - two count x count matrices with complex numbers
  std::vector<std::complex<double>> in(2 * count * count, 0.0);
  std::vector<std::complex<double>> out(count * count, 0.0);

  // Create identity matrices for testing
  for (size_t i = 0; i < count; i++) {
    in[(i * count) + i] = std::complex<double>(1.0, 0.0);                    // First matrix (identity)
    in[(count * count) + (i * count) + i] = std::complex<double>(1.0, 0.0);  // Second matrix (identity)
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_tbb::SparseMatrixComplexCRS test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  // Check result (identity * identity = identity)
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < count; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * count) + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[(i * count) + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * count) + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[(i * count) + j].imag(), 0.0, 1e-10);
      }
    }
  }
}