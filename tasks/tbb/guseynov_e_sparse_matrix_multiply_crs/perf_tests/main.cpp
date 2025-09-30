#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/guseynov_e_sparse_matrix_multiply_crs/include/ops_tbb.hpp"

namespace {
struct MatrixParams {
  int rows;
  int cols;
  double density;
  int seed;
};

guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix GenerateRandomMatrix(const MatrixParams &params) {
  std::mt19937 gen;
  gen.seed(params.seed);
  std::uniform_real_distribution<double> random(-2.0, 2.0);
  std::bernoulli_distribution bernoulli(params.density);

  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix result;
  result.n_rows = params.rows;
  result.n_cols = params.cols;
  result.pointer.assign(result.n_rows + 1, 0);
  std::vector<std::vector<std::pair<int, double>>> temp(result.n_rows);
  for (int i = 0; i < params.rows; i++) {
    for (int j = 0; j < params.cols; j++) {
      if (bernoulli(gen)) {
        double val(random(gen));
        temp[i].emplace_back(j, val);
      }
    }
  }
  for (int i = 0; i < result.n_rows; i++) {
    result.pointer[i + 1] = result.pointer[i];
    for (auto &j : temp[i]) {
      result.col_indexes.push_back(j.first);
      result.non_zero_values.push_back(j.second);
      result.pointer[i + 1]++;
    }
  }

  return result;
}

guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix T(const guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix &m) {
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix temp_matrix;
  temp_matrix.n_rows = m.n_cols;
  temp_matrix.n_cols = m.n_rows;
  temp_matrix.pointer.assign(temp_matrix.n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(temp_matrix.n_rows);
  for (int i = 0; i < m.n_rows; i++) {
    for (int k = m.pointer[i]; k < m.pointer[i + 1]; k++) {
      int j = m.col_indexes[k];
      temp[j].emplace_back(i, m.non_zero_values[k]);
    }
  }

  for (int i = 0; i < temp_matrix.n_rows; i++) {
    temp_matrix.pointer[i + 1] = temp_matrix.pointer[i];
    for (auto &j : temp[i]) {
      temp_matrix.col_indexes.push_back(j.first);
      temp_matrix.non_zero_values.push_back(j.second);
      temp_matrix.pointer[i + 1]++;
    }
  }

  return temp_matrix;
}

guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix MultiplyCRSSeq(
    const guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix &a_mat,
    guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix b_mat) {
  // транспонируем локальную копию B
  b_mat = T(b_mat);

  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix result;
  result.n_rows = a_mat.n_rows;
  result.n_cols = b_mat.n_rows;
  result.pointer.assign(result.n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(result.n_rows);

  for (int i = 0; i < result.n_rows; i++) {
    for (int j = 0; j < b_mat.n_rows; j++) {
      double sum = 0.0;
      for (int k_a = a_mat.pointer[i]; k_a < a_mat.pointer[i + 1]; k_a++) {
        for (int k_b = b_mat.pointer[j]; k_b < b_mat.pointer[j + 1]; k_b++) {
          if (a_mat.col_indexes[k_a] == b_mat.col_indexes[k_b]) {
            sum += a_mat.non_zero_values[k_a] * b_mat.non_zero_values[k_b];
          }
        }
      }
      if (std::abs(sum) > 1e-12) {  // отсекаем нули
        temp[i].emplace_back(j, sum);
      }
    }
  }

  for (int i = 0; i < result.n_rows; i++) {
    result.pointer[i + 1] = result.pointer[i];
    for (auto &pr : temp[i]) {
      result.col_indexes.push_back(pr.first);
      result.non_zero_values.push_back(pr.second);
      result.pointer[i + 1]++;
    }
  }

  return result;
}

void CompareRows(const std::vector<std::pair<int, double>> &row_r, const std::vector<std::pair<int, double>> &row_s,
                 double eps) {
  ASSERT_EQ(row_r.size(), row_s.size());
  for (size_t j = 0; j < row_r.size(); ++j) {
    EXPECT_EQ(row_r[j].first, row_s[j].first);
    EXPECT_NEAR(row_r[j].second, row_s[j].second, eps);
  }
}

void CompareCRSMatricesUnordered(const guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix &result,
                                 const guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix &expected,
                                 double eps = 1e-9) {
  ASSERT_EQ(result.n_rows, expected.n_rows);
  ASSERT_EQ(result.n_cols, expected.n_cols);
  ASSERT_EQ(result.pointer, expected.pointer);

  for (int i = 0; i < result.n_rows; i++) {
    int start_r = result.pointer[i];
    int end_r = result.pointer[i + 1];
    int start_s = expected.pointer[i];
    int end_s = expected.pointer[i + 1];

    ASSERT_EQ(end_r - start_r, end_s - start_s);

    std::vector<std::pair<int, double>> row_r;
    std::vector<std::pair<int, double>> row_s;
    for (int k = start_r; k < end_r; ++k) {
      row_r.emplace_back(result.col_indexes[k], result.non_zero_values[k]);
    }

    for (int k = start_s; k < end_s; ++k) {
      row_s.emplace_back(expected.col_indexes[k], expected.non_zero_values[k]);
    }

    std::ranges::sort(row_r);
    std::ranges::sort(row_s);

    CompareRows(row_r, row_s, eps);
  }
}
}  // namespace

TEST(guseynov_e_sparse_matrix_multiply_crs_tbb, test_pipeline_run) {
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix a =
      GenerateRandomMatrix({.rows = 200, .cols = 200, .density = 0.6, .seed = 1993});
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix b =
      GenerateRandomMatrix({.rows = 200, .cols = 200, .density = 0.6, .seed = 4325});
  auto b_seq = b;
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test_task = std::make_shared<guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto seq_result = MultiplyCRSSeq(a, b_seq);
  CompareCRSMatricesUnordered(result, seq_result);
}

TEST(guseynov_e_sparse_matrix_multiply_crs_tbb, test_task_run) {
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix a =
      GenerateRandomMatrix({.rows = 200, .cols = 200, .density = 0.6, .seed = 1993});
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix b =
      GenerateRandomMatrix({.rows = 200, .cols = 200, .density = 0.6, .seed = 4325});
  auto b_seq = b;
  guseynov_e_sparse_matrix_multiply_crs_tbb::CRSMatrix result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test_task = std::make_shared<guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto seq_result = MultiplyCRSSeq(a, b_seq);
  CompareCRSMatricesUnordered(result, seq_result);
}