#include "seq/muradov_k_histogram_stretch/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace muradov_k_histogram_stretch_seq {

bool HistogramStretchSequential::PreProcessingImpl() {
  // Считываем входной вектор пикселей
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_image_.assign(in_ptr, in_ptr + input_size);
  // Готовим выходной буфер
  unsigned int output_size = task_data->outputs_count[0];
  output_image_.assign(output_size, 0);
  return true;
}

bool HistogramStretchSequential::ValidationImpl() {
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) return false;
  if (task_data->inputs_count[0] == 0) return false;
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) return false;
  // Проверяем диапазон пикселей (допускаем 0..255)
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; ++i) {
    if (in_ptr[i] < 0 || in_ptr[i] > 255) return false;
  }
  return true;
}

bool HistogramStretchSequential::RunImpl() {
  if (input_image_.empty()) return false;
  auto mm = std::minmax_element(input_image_.begin(), input_image_.end());
  min_val_ = *mm.first;
  max_val_ = *mm.second;
  if (min_val_ == max_val_) {
    // Все пиксели одинаковые -> приводим к нулю
    std::fill(output_image_.begin(), output_image_.end(), 0);
    return true;
  }
  const int range = max_val_ - min_val_;
  for (size_t i = 0; i < input_image_.size(); ++i) {
    int stretched = (input_image_[i] - min_val_) * 255 / range;  // целочисленное масштабирование
    if (stretched < 0) stretched = 0;
    if (stretched > 255) stretched = 255;
    output_image_[i] = stretched;
  }
  return true;
}

bool HistogramStretchSequential::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_image_.size(); ++i) {
    out_ptr[i] = output_image_[i];
  }
  return true;
}

}  // namespace muradov_k_histogram_stretch_seq
