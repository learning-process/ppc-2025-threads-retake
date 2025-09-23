#include "tbb/koshkin_n_convex_hull_binary_image/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);
  unsigned int size = static_cast<unsigned int>(width_) * static_cast<unsigned int>(height_);

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

long long koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::Cross(const Pt& a, const Pt& b, const Pt& c) {
  long long bax = static_cast<long long>(b.first) - a.first;
  long long bay = static_cast<long long>(b.second) - a.second;
  long long cax = static_cast<long long>(c.first) - a.first;
  long long cay = static_cast<long long>(c.second) - a.second;
  return (bax * cay) - (bay * cax);
}

long long koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::Dist2(const Pt& a, const Pt& b) {
  long long dx = static_cast<long long>(a.first) - b.first;
  long long dy = static_cast<long long>(a.second) - b.second;
  return (dx * dx) + (dy * dy);
}

inline bool koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::IsBorderPixel(const std::vector<int>& in,
                                                                                         int width, int height, int x,
                                                                                         int y) {
  int row_off = y * width;
  if (in[row_off + x] != 1) {
    return false;
  }

  static const int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static const int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    return true;
  }

  for (int k = 0; k < 8; ++k) {
    int nx = x + kDx[k];
    int ny = y + kDy[k];
    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
      return true;
    }
    if (in[(ny * width) + nx] == 0) {
      return true;
    }
  }
  return false;
}

void koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::FindPoints() {
  points_.clear();

  if (width_ <= 0 || height_ <= 0) {
    return;
  }

  const int w = width_;
  const int h = height_;

  tbb::concurrent_vector<Pt> cv;
  cv.reserve(static_cast<std::size_t>(std::min(1024, w * h)));

  tbb::parallel_for(tbb::blocked_range<int>(0, h), [&](const tbb::blocked_range<int>& range) {
    for (int y = range.begin(); y < range.end(); ++y) {
      int row_off = y * w;
      for (int x = 0; x < w; ++x) {
        if (input_[row_off + x] != 1) {
          continue;
        }
        if (IsBorderPixel(input_, w, h, x, y)) {
          cv.push_back(Pt{x, y});
        }
      }
    }
  });

  points_.assign(cv.begin(), cv.end());
}

bool koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::RunImpl() {
  FindPoints();

  if (points_.empty()) {
    output_.clear();
    return true;
  }

  // (parallel sort)
  tbb::parallel_sort(points_.begin(), points_.end(), [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });

  auto uniq_end = std::unique(points_.begin(), points_.end());
  points_.erase(uniq_end, points_.end());

  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }

  // parallel_reduce
  Pt initial_pivot = points_[0];
  auto pivot_pair = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, points_.size()), std::make_pair(initial_pivot, true),
      [&](const tbb::blocked_range<std::size_t>& r, std::pair<Pt, bool> local) -> std::pair<Pt, bool> {
        Pt best = local.second ? local.first : Pt{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
        bool inited = local.second;
        for (std::size_t i = r.begin(); i < r.end(); ++i) {
          const Pt& p = points_[i];
          if (!inited || p.second < best.second || (p.second == best.second && p.first < best.first)) {
            best = p;
            inited = true;
          }
        }
        return std::make_pair(best, inited);
      },
      [&](const std::pair<Pt, bool>& a, const std::pair<Pt, bool>& b) -> std::pair<Pt, bool> {
        if (!a.second) {
          return b;
        }
        if (!b.second) {
          return a;
        }
        const Pt &pa = a.first, &pb = b.first;
        if (pa.second < pb.second || (pa.second == pb.second && pa.first < pb.first)) {
          return a;
        }
        return b;
      });
  Pt pivot = pivot_pair.first;

  tbb::parallel_sort(points_.begin(), points_.end(), [&](const Pt& a, const Pt& b) {
    if (a == b) {
      return false;
    }
    long long cr = Cross(pivot, a, b);
    if (cr != 0) {
      return cr > 0;
    }
    return Dist2(pivot, a) < Dist2(pivot, b);
  });

  // Graham scan (sequential)
  std::vector<Pt> hull;
  hull.reserve(points_.size());
  for (const Pt& p : points_) {
    while (hull.size() >= 2) {
      Pt q = hull[hull.size() - 2];
      Pt r = hull[hull.size() - 1];
      long long cr = Cross(q, r, p);
      if (cr <= 0) {
        hull.pop_back();
      } else {
        break;
      }
    }
    hull.push_back(p);
  }

  if (hull.size() == 1 && points_.size() > 1) {
    hull.push_back(points_.back());
  }

  output_ = std::move(hull);
  return true;
}

bool koshkin_n_convex_hull_binary_image_tbb::ConvexHullBinaryImage::PostProcessingImpl() {
  std::ranges::sort(output_, [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });

  Pt* dest = reinterpret_cast<Pt*>(task_data->outputs[0]);
  std::copy_n(output_.begin(), output_.size(), dest);
  task_data->outputs_count[0] = static_cast<int>(output_.size());
  return true;
}