#include "radix_double_seq.hpp"

#include <algorithm>
#include <array>

namespace yasakova_t_sort_seq {

bool SortTaskSequential::ValidationImpl() {
    if (!task_data) return false;
    if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) return false;
    if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) return false;
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) return false;
    return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskSequential::PreProcessingImpl() {
    const size_t count = static_cast<size_t>(task_data->inputs_count[0]);
    const auto* in_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
    input_.assign(in_ptr, in_ptr + count);
    output_.assign(count, 0.0);
    return true;
}

bool SortTaskSequential::RunImpl() {
    output_ = input_;
    radix_sort_double_seq(output_);
    return true;
}

bool SortTaskSequential::PostProcessingImpl() {
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(output_.begin(), output_.end(), out_ptr);
    return true;
}

void radix_sort_double_seq(std::vector<double>& a) {
    // Move NaNs to the tail while keeping sortable values separate
    std::vector<double> nonnan;
    nonnan.reserve(a.size());
    std::vector<double> nans;
    nans.reserve(16);
    for (double x : a) {
        (is_nan(x) ? nans : nonnan).push_back(x);
    }

    const size_t n = nonnan.size();
    if (n <= 1) {
        a.assign(nonnan.begin(), nonnan.end());
        a.insert(a.end(), nans.begin(), nans.end());
        return;
    }

    std::vector<uint64_t> keys(n);
    for (size_t i = 0; i < n; ++i) keys[i] = to_key(nonnan[i]);

    std::vector<uint64_t> buf(n);
    for (int pass = 0; pass < 8; ++pass) {
        std::array<size_t, 256> cnt{};
        const int shift = pass * 8;

        for (size_t i = 0; i < n; ++i) {
            ++cnt[static_cast<uint8_t>(keys[i] >> shift)];
        }

        size_t sum = 0;
        for (int b = 0; b < 256; ++b) {
            size_t c = cnt[b];
            cnt[b] = sum;
            sum += c;
        }

        for (size_t i = 0; i < n; ++i) {
            uint8_t b = static_cast<uint8_t>(keys[i] >> shift);
            buf[cnt[b]++] = keys[i];
        }
        keys.swap(buf);
    }

    for (size_t i = 0; i < n; ++i) nonnan[i] = from_key(keys[i]);

    a.assign(nonnan.begin(), nonnan.end());
    a.insert(a.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_seq
