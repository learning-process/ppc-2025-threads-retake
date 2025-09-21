#pragma once
#include <vector>

namespace bobylev_m_radix_double_omp {
// Сортирует по возрастанию
void parallel_radix_sort_double_with_simple_merge(std::vector<double>& a, int blocks = 0);
}  // namespace bobylev_m_radix_double_omp
