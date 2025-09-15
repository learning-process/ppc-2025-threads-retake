#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

bool tarakanov_d_fox_algorithm_seq::TaskSequential::PreProcessingImpl() {
  // Init value for input and output
  return true;
}

bool tarakanov_d_fox_algorithm_seq::TaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return true;
}

bool tarakanov_d_fox_algorithm_seq::TaskSequential::RunImpl() {
  // Multiply matrices
  return true;
}

bool tarakanov_d_fox_algorithm_seq::TaskSequential::PostProcessingImpl() {
  return true;
}
