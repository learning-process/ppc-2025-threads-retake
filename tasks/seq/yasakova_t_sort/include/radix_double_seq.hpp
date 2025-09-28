#pragma once
#include <cstdint>
#include <vector>
#include <bit>
#include <cmath>

namespace yasakova_t_sort_seq {

inline bool is_nan(double x) { return std::isnan(x); }

// double -> ключ с тотальным порядком (IEEE-754)
inline uint64_t to_key(double x) {
    uint64_t b = std::bit_cast<uint64_t>(x);
    if ((b >> 63) == 0) return b ^ 0x8000'0000'0000'0000ull;
    return ~b;
}
// ключ -> double (обратное преобразование)
inline double from_key(uint64_t k) {
    if ((k >> 63) != 0) {
        uint64_t b = k ^ 0x8000'0000'0000'0000ull;
        return std::bit_cast<double>(b);
    } else {
        uint64_t b = ~k;
        return std::bit_cast<double>(b);
    }
}

// Стабильная LSD-radix сортировка по 8 байтам
void radix_sort_double_seq(std::vector<double>& a);

} // namespace yasakova_t_sort_seq
