#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

TEST(Ivashchuk_V_sparse_matrix_seq, test_identity_matrix) {
  constexpr size_t kCount = 50;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
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

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
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

TEST(Ivashchuk_V_sparse_matrix_seq, test_complex_multiplication) {
  constexpr size_t kCount = 30;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
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

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
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

TEST(Ivashchuk_V_sparse_matrix_seq, test_zero_matrix) {
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

  for (size_t i = 0; i < kCount * kCount; i++) {
    EXPECT_NEAR(out[i].real(), 0.0, 1e-10);
    EXPECT_NEAR(out[i].imag(), 0.0, 1e-10);
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, test_diagonal_matrices) {
  constexpr size_t kCount = 25;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in1[(i * kCount) + i] = std::complex<double>(i + 1, 0.5 * i);
    in2[(i * kCount) + i] = std::complex<double>(kCount - i, -0.3 * i);
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

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
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

TEST(Ivashchuk_V_sparse_matrix_seq, test_sparse_random_matrices) {
  constexpr size_t kCount = 15;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);
  std::vector<std::complex<double>> expected(kCount * kCount, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-2.0, 2.0);

  // Создаем разреженные матрицы с несколькими ненулевыми элементами
  for (size_t i = 0; i < kCount; i++) {
    // Диагональные элементы
    in1[i * kCount + i] = std::complex<double>(dis(gen), dis(gen));
    in2[i * kCount + i] = std::complex<double>(dis(gen), dis(gen));

    // Несколько случайных элементов
    if (i < kCount - 1) {
      in1[i * kCount + (i + 1)] = std::complex<double>(dis(gen), dis(gen));
      in2[(i + 1) * kCount + i] = std::complex<double>(dis(gen), dis(gen));
    }
  }

  // Вычисляем ожидаемый результат напрямую
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      for (size_t k = 0; k < kCount; k++) {
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

  for (size_t i = 0; i < kCount * kCount; i++) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, test_single_element_matrices) {
  constexpr size_t kCount = 3;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  // Матрица A: элемент [0][2] = (2.5, -1.0) - строка 0, столбец 2
  in1[2] = std::complex<double>(2.5, -1.0);
  // Матрица B: элемент [2][1] = (-1.5, 2.0) - строка 2, столбец 1
  in2[7] = std::complex<double>(-1.5, 2.0);

  std::cout << "=== DEBUG TEST SINGLE ELEMENT ===" << std::endl;
  std::cout << "Matrix A (3x3):" << std::endl;
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      std::cout << "(" << in1[i * kCount + j].real() << ", " << in1[i * kCount + j].imag() << ") ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix B (3x3):" << std::endl;
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      std::cout << "(" << in2[i * kCount + j].real() << ", " << in2[i * kCount + j].imag() << ") ";
    }
    std::cout << std::endl;
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

  std::cout << "Result matrix (3x3):" << std::endl;
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      std::cout << "(" << out[i * kCount + j].real() << ", " << out[i * kCount + j].imag() << ") ";
    }
    std::cout << std::endl;
  }

  // Ожидаемый результат: A[0][2] * B[2][1] = (2.5, -1.0) * (-1.5, 2.0)
  // Расчет: (2.5 * -1.5 - (-1.0) * 2.0) + i(2.5 * 2.0 + (-1.0) * -1.5)
  // = (-3.75 + 2.0) + i(5.0 + 1.5) = (-1.75) + i(6.5)
  // Результат должен быть в C[0][1]
  std::complex<double> expected(-1.75, 6.5);
  std::cout << "Expected result at [0][1]: (" << expected.real() << ", " << expected.imag() << ")" << std::endl;

  // Проверяем результат
  bool found_expected = false;
  for (size_t i = 0; i < kCount * kCount; i++) {
    if (std::abs(out[i].real()) > 1e-10 || std::abs(out[i].imag()) > 1e-10) {
      std::cout << "Non-zero element at [" << i / kCount << "][" << i % kCount << "]: (" << out[i].real() << ", "
                << out[i].imag() << ")" << std::endl;
      if (std::abs(out[i].real() - expected.real()) < 1e-10 && std::abs(out[i].imag() - expected.imag()) < 1e-10) {
        found_expected = true;
      }
    }
  }

  EXPECT_TRUE(found_expected) << "Expected result not found in output matrix";
  std::cout << "=== END DEBUG ===" << std::endl;
}

TEST(Ivashchuk_V_sparse_matrix_seq, test_validation_failure) {
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

  // Должно вернуть false из-за разных размеров матриц
  ASSERT_EQ(test_task_sequential->Validation(), false);
}

TEST(Ivashchuk_V_sparse_matrix_seq, test_different_sizes) {
  constexpr size_t kCount1 = 8;
  constexpr size_t kCount2 = 12;
  constexpr size_t kCount3 = 16;

  std::vector<std::complex<double>> in1(kCount1 * kCount1, 0);
  std::vector<std::complex<double>> in2(kCount2 * kCount2, 0);
  std::vector<std::complex<double>> out(kCount3 * kCount3, 0);

  // Заполняем матрицы
  for (size_t i = 0; i < kCount1; i++) {
    in1[i * kCount1 + i] = std::complex<double>(1.0, 0.0);
  }
  for (size_t i = 0; i < kCount2; i++) {
    in2[i * kCount2 + i] = std::complex<double>(1.0, 0.0);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  // Должно провалить валидацию из-за разных размеров
  ASSERT_EQ(test_task_sequential->Validation(), false);
}