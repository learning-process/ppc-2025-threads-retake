#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

TEST(Ivashchuk_V_sparse_matrix_seq, TestIdentityMatrix) {
  constexpr size_t kCount = 50;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
    in2[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      if (i == j) {
        EXPECT_NEAR(out[i * kCount + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[i * kCount + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestComplexMultiplication) {
  constexpr size_t kCount = 30;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[(i * kCount) + i] = std::complex<double>(2.0, 1.0);
    in2[(i * kCount) + i] = std::complex<double>(3.0, -1.0);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      if (i == j) {
        EXPECT_NEAR(out[i * kCount + j].real(), 7.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 1.0, 1e-10);
      } else {
        EXPECT_NEAR(out[i * kCount + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestZeroMatrix) {
  constexpr size_t kCount = 20;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i].real(), 0.0, 1e-10);
    EXPECT_NEAR(out[i].imag(), 0.0, 1e-10);
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestDiagonalMatrices) {
  constexpr size_t kCount = 25;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[(i * kCount) + i] = std::complex<double>(static_cast<double>(i) + 1.0, 0.5 * static_cast<double>(i));
    in2[(i * kCount) + i] =
        std::complex<double>(static_cast<double>(kCount) - static_cast<double>(i), -0.3 * static_cast<double>(i));
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      if (i == j) {
        std::complex<double> expected = in1[i * kCount + i] * in2[i * kCount + i];
        EXPECT_NEAR(out[i * kCount + j].real(), expected.real(), 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), expected.imag(), 1e-10);
      } else {
        EXPECT_NEAR(out[i * kCount + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestSparseRandomMatrices) {
  constexpr size_t kCount = 15;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);
  std::vector<std::complex<double>> expected(kCount * kCount, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-2.0, 2.0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[i * kCount + i] = std::complex<double>(dis(gen), dis(gen));
    in2[i * kCount + i] = std::complex<double>(dis(gen), dis(gen));

    if (i < kCount - 1) {
      in1[i * kCount + (i + 1)] = std::complex<double>(dis(gen), dis(gen));
      in2[(i + 1) * kCount + i] = std::complex<double>(dis(gen), dis(gen));
    }
  }

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      for (size_t k = 0; k < kCount; ++k) {
        expected[i * kCount + j] += in1[i * kCount + k] * in2[k * kCount + j];
      }
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestSingleElementMatrices) {
  constexpr size_t kCount = 2;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  in1[0] = std::complex<double>(2.0, 1.0);
  in2[0] = std::complex<double>(3.0, -1.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  std::complex<double> expected = in1[0] * in2[0];

  EXPECT_NEAR(out[0].real(), expected.real(), 1e-10);
  EXPECT_NEAR(out[0].imag(), expected.imag(), 1e-10);

  for (size_t i = 1; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i].real(), 0.0, 1e-10);
    EXPECT_NEAR(out[i].imag(), 0.0, 1e-10);
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestValidationFailure) {
  constexpr size_t kCount1 = 10;
  constexpr size_t kCount2 = 15;

  std::vector<std::complex<double>> in1(kCount1 * kCount1, 0);
  std::vector<std::complex<double>> in2(kCount2 * kCount2, 0);
  std::vector<std::complex<double>> out(kCount1 * kCount1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  ASSERT_EQ(test_task_sequential->Validation(), false);
}