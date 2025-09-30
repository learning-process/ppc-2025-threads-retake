#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/Ivashchuk_V_mult_crs/include/ops_omp.hpp"

// Forward declarations
void TestMultiply3x3();
void TestMultiplyComplexNumbers();
void TestSparseMatrixMultiplication();
void TestMultiplyZeroMatrix();
void TestMultiplyDiagonalMatrices();
void TestMultiplyRectangularMatrices();
void TestMultiplyWithImaginaryNumbers();

void TestMultiply3x3() {
  constexpr int kRows1 = 3;
  constexpr int kCols1 = 3;
  constexpr int kRows2 = 3;
  constexpr int kCols2 = 3;

  // Create data - identity matrices
  std::vector<std::complex<double>> in1(kRows1 * kCols1, {0.0, 0.0});
  std::vector<std::complex<double>> in2(kRows2 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});

  // Identity matrix
  for (int i = 0; i < kRows1; i++) {
    in1[(i * kCols1) + i] = {1.0, 0.0};
    in2[(i * kCols2) + i] = {1.0, 0.0};
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  // Matrix dimensions
  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result (should be identity matrix)
  for (int i = 0; i < kRows1; i++) {
    for (int j = 0; j < kCols2; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * kCols2) + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[(i * kCols2) + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * kCols2) + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[(i * kCols2) + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_3x3) { TestMultiply3x3(); }

void TestMultiplyComplexNumbers() {
  constexpr int kRows1 = 2;
  constexpr int kCols1 = 2;
  constexpr int kRows2 = 2;
  constexpr int kCols2 = 2;

  std::vector<std::complex<double>> in1 = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

  std::vector<std::complex<double>> in2 = {{2.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {2.0, 2.0}};

  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> expected = {{-3.0, 11.0}, {-8.0, 10.0}, {-7.0, 31.0}, {-12.0, 38.0}};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_complex_numbers) { TestMultiplyComplexNumbers(); }

void TestSparseMatrixMultiplication() {
  constexpr int kRows1 = 3;
  constexpr int kCols1 = 4;
  constexpr int kRows2 = 4;
  constexpr int kCols2 = 2;

  // Sparse matrices with mostly zeros
  std::vector<std::complex<double>> in1 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {4.0, 0.0}, {0.0, 0.0}, {2.0, 0.0},
                                           {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}, {0.0, 0.0}};

  std::vector<std::complex<double>> in2 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {2.0, 0.0},
                                           {3.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {4.0, 0.0}};

  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> expected = {{1.0, 0.0}, {16.0, 0.0}, {0.0, 0.0},
                                                {4.0, 0.0}, {9.0, 0.0},  {0.0, 0.0}};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_sparse_matrix_multiplication) { TestSparseMatrixMultiplication(); }

void TestMultiplyZeroMatrix() {
  constexpr int kRows1 = 3;
  constexpr int kCols1 = 3;
  constexpr int kRows2 = 3;
  constexpr int kCols2 = 3;

  // Create zero matrices
  std::vector<std::complex<double>> in1(kRows1 * kCols1, {0.0, 0.0});
  std::vector<std::complex<double>> in2(kRows2 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result (should be zero matrix)
  for (int i = 0; i < kRows1; i++) {
    for (int j = 0; j < kCols2; j++) {
      EXPECT_NEAR(out[(i * kCols2) + j].real(), 0.0, 1e-10);
      EXPECT_NEAR(out[(i * kCols2) + j].imag(), 0.0, 1e-10);
    }
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_zero_matrix) { TestMultiplyZeroMatrix(); }

void TestMultiplyDiagonalMatrices() {
  constexpr int kRows1 = 4;
  constexpr int kCols1 = 4;
  constexpr int kRows2 = 4;
  constexpr int kCols2 = 4;

  std::vector<std::complex<double>> in1(kRows1 * kCols1, {0.0, 0.0});
  std::vector<std::complex<double>> in2(kRows2 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});

  // Diagonal matrices with different values
  for (int i = 0; i < kRows1; i++) {
    in1[(i * kCols1) + i] = {static_cast<double>(i + 1), 0.0};
    in2[(i * kCols2) + i] = {static_cast<double>(i + 2), 0.0};
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result (should be diagonal matrix with products)
  for (int i = 0; i < kRows1; i++) {
    for (int j = 0; j < kCols2; j++) {
      if (i == j) {
        double expected = static_cast<double>((i + 1) * (i + 2));
        EXPECT_NEAR(out[(i * kCols2) + j].real(), expected, 1e-10);
        EXPECT_NEAR(out[(i * kCols2) + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * kCols2) + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[(i * kCols2) + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_diagonal_matrices) { TestMultiplyDiagonalMatrices(); }

void TestMultiplyRectangularMatrices() {
  constexpr int kRows1 = 2;
  constexpr int kCols1 = 3;
  constexpr int kRows2 = 3;
  constexpr int kCols2 = 2;

  std::vector<std::complex<double>> in1 = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}};
  std::vector<std::complex<double>> in2 = {{7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}};
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> expected = {{58.0, 0.0}, {64.0, 0.0}, {139.0, 0.0}, {154.0, 0.0}};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_rectangular_matrices) { TestMultiplyRectangularMatrices(); }

void TestMultiplyWithImaginaryNumbers() {
  constexpr int kRows1 = 2;
  constexpr int kCols1 = 2;
  constexpr int kRows2 = 2;
  constexpr int kCols2 = 2;

  std::vector<std::complex<double>> in1 = {{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}};
  std::vector<std::complex<double>> in2 = {{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}};
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});
  std::vector<std::complex<double>> expected = {{-5.0, 0.0}, {-8.0, 0.0}, {-11.0, 0.0}, {-20.0, 0.0}};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check result
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i].real(), expected[i].real(), 1e-10);
    EXPECT_NEAR(out[i].imag(), expected[i].imag(), 1e-10);
  }
}

TEST(Ivashchuk_V_mult_crs_omp, test_multiply_with_imaginary_numbers) { TestMultiplyWithImaginaryNumbers(); }