#include "radix_double_seq.hpp"
#include <array>
#include <algorithm>

namespace yasakova_t_sort_seq {

void radix_sort_double_seq(std::vector<double>& a) {
    // NaN отправляем в хвост
    std::vector<double> nonnan; nonnan.reserve(a.size());
    std::vector<double> nans;   nans.reserve(16);
    for (double x : a) (is_nan(x) ? nans : nonnan).push_back(x);

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
        std::array<size_t, 256> cnt{}; // zero-init
        const int shift = pass * 8;

        for (size_t i = 0; i < n; ++i) ++cnt[ static_cast<uint8_t>(keys[i] >> shift) ];

        size_t sum = 0;
        for (int b = 0; b < 256; ++b) { size_t c = cnt[b]; cnt[b] = sum; sum += c; }

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

} // namespace yasakova_t_sort_seq
