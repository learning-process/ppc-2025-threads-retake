#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bobylev_m_radix_double_omp {

static inline uint64_t dbl_bits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}
static inline double bits_dbl(uint64_t u) {
  double x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

static inline uint64_t double_to_key(double x) {
  uint64_t u = dbl_bits(x);
  if (u >> 63)
    return ~u;  // отрицательное
  else
    return u ^ 0x8000000000000000ULL;  // неотрицательное
}
static inline double key_to_double(uint64_t k) {
  uint64_t u;
  if (k & 0x8000000000000000ULL)
    u = k ^ 0x8000000000000000ULL;  // было >=0
  else
    u = ~k;  // было <0               
  return bits_dbl(u);
}

// Один стабильный проход LSD по одному байту (0..7) с параллельной раскладкой
static void radix_pass_parallel(const uint64_t* in, uint64_t* out, size_t n, int byte_index) {
  constexpr int B = 256;
  const int SHIFT = byte_index * 8;

  int T = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    { T = omp_get_num_threads(); }
  }
#endif

  std::vector<std::array<size_t, B>> local_counts(T);
  for (int t = 0; t < T; ++t) local_counts[t].fill(0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    size_t start = (n * (size_t)tid) / T;
    size_t finish = (n * (size_t)(tid + 1)) / T;
    auto& cnt = local_counts[tid];

    for (size_t i = start; i < finish; ++i) {
      uint8_t b = (uint8_t)((in[i] >> SHIFT) & 0xFFu);
      cnt[b]++;
    }
  }

  std::array<size_t, B> global_counts{};
  global_counts.fill(0);
  for (int b = 0; b < B; ++b) {
    size_t s = 0;
    for (int t = 0; t < T; ++t) s += local_counts[t][b];
    global_counts[b] = s;
  }

  std::array<size_t, B> global_prefix{};
  {
    size_t acc = 0;
    for (int b = 0; b < B; ++b) {
      global_prefix[b] = acc;
      acc += global_counts[b];
    }
  }

  std::vector<std::array<size_t, B>> thread_starts(T);
  for (int b = 0; b < B; ++b) {
    size_t base = global_prefix[b];
    for (int t = 0; t < T; ++t) {
      thread_starts[t][b] = base;
      base += local_counts[t][b];
    }
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    size_t start = (n * (size_t)tid) / T;
    size_t finish = (n * (size_t)(tid + 1)) / T;

    std::array<size_t, B> cursor{};
    for (int b = 0; b < B; ++b) cursor[b] = thread_starts[tid][b];

    for (size_t i = start; i < finish; ++i) {
      uint8_t b = (uint8_t)((in[i] >> SHIFT) & 0xFFu);
      out[cursor[b]++] = in[i];
    }
  }
}

// Полный LSD radix по всем 8 байтам (внутри — параллельные проходы)
static void radix_sort_keys_parallel(uint64_t* a, size_t n) {
  if (n <= 1) return;
  std::vector<uint64_t> buf(n);
  uint64_t* src = a;
  uint64_t* dst = buf.data();

  for (int byte = 0; byte < 8; ++byte) {
    radix_pass_parallel(src, dst, n, byte);
    std::swap(src, dst);
  }
  if (src != a) {
    std::memcpy(a, src, n * sizeof(uint64_t));
  }
}

// Слияние двух отсортированных отрезков ключей: [l, m) и [m, r)
static void merge_two(const uint64_t* in, uint64_t* out, size_t l, size_t m, size_t r) {
  size_t i = l, j = m, k = l;
  while (i < m && j < r) {
    if (in[i] <= in[j])
      out[k++] = in[i++];
    else
      out[k++] = in[j++];
  }
  while (i < m) out[k++] = in[i++];
  while (j < r) out[k++] = in[j++];
}

// Параллельная по блокам (radix на блоках + простые слияния)
void parallel_radix_sort_double_with_simple_merge(std::vector<double>& a, int blocks) {
  const size_t n = a.size();
  if (n <= 1) return;

  int T = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    { T = omp_get_num_threads(); }
  }
#endif
  if (blocks <= 0) blocks = std::max(1, T);
  blocks = std::min<int>(blocks, (int)n);

  std::vector<uint64_t> keys(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long i = 0; i < (long long)n; ++i) {
    keys[i] = double_to_key(a[i]);
  }

  std::vector<size_t> starts(blocks + 1, 0);
  for (int b = 0; b < blocks; ++b) starts[b] = (n * (size_t)b) / blocks;
  starts[blocks] = n;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < blocks; ++b) {
    size_t l = starts[b];
    size_t r = starts[b + 1];
    if (r > l) {
      radix_sort_keys_parallel(keys.data() + l, r - l);
    }
  }

  std::vector<uint64_t> tmp(n);
  for (int width = 1; width < blocks; width *= 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < blocks; i += 2 * width) {
      size_t l = starts[i];
      size_t m = (i + width <= blocks) ? starts[i + width] : starts[i];
      size_t r = (i + 2 * width <= blocks) ? starts[i + 2 * width] : starts[std::min(i + width, blocks)];
      if (m > l && r > m) {
        merge_two(keys.data(), tmp.data(), l, m, r);
      } else {
        if (r > l) std::memcpy(tmp.data() + l, keys.data() + l, (r - l) * sizeof(uint64_t));
          }
      }
    }
    std::swap(keys, tmp);

    int newBlocks = 0;
    for (int i = 0; i < blocks; i += 2 * width) {
      starts[newBlocks++] = starts[i];
    }
    starts[newBlocks] = n;
    blocks = newBlocks;
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long i = 0; i < (long long)n; ++i) {
    a[i] = key_to_double(keys[i]);
  }
}

}  // namespace bobylev_m_radix_double_omp
