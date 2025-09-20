#include "seq/koshkin_n_convex_hull_binary_image/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);
  unsigned int size = width_ * height_;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

long long koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::Cross(const Pt& a, const Pt& b, const Pt& c) {
  // cross product (b - a) x (c - a)
  long long bax = static_cast<long long>(b.first) - a.first;
  long long bay = static_cast<long long>(b.second) - a.second;
  long long cax = static_cast<long long>(c.first) - a.first;
  long long cay = static_cast<long long>(c.second) - a.second;
  return (bax * cay) - (bay * cax);
}

long long koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::Dist2(const Pt& a, const Pt& b) {
  long long dx = static_cast<long long>(a.first) - b.first;
  long long dy = static_cast<long long>(a.second) - b.second;
  return (dx * dx) + (dy * dy);
}

inline bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::IsBorderPixel(const std::vector<int>& in,
                                                                                         int width, int height, int x,
                                                                                         int y) {
  int row_off = y * width;
  if (in[row_off + x] != 1) {
    return false;
  }

  static const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    return true;
  }

  for (int k = 0; k < 8; ++k) {
    int nx = x + dx[k];
    int ny = y + dy[k];
    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
      return true;
    }
    if (in[(ny * width) + nx] == 0) {
      return true;
    }
  }
  return false;
}

void koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::FindPoints() {
  points_.clear();

  if (width_ <= 0 || height_ <= 0) return;

  points_.reserve(256);
  for (int y = 0; y < height_; ++y) {
    int row_off = y * width_;
    for (int x = 0; x < width_; ++x) {
      if (input_[row_off + x] != 1) {
        continue;
      }
      if (IsBorderPixel(input_, width_, height_, x, y)) {
        points_.emplace_back(x, y);
      }
    }
  }
}

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::RunImpl() {
  FindPoints();

  if (points_.empty()) {
    output_.clear();
    return true;
  }

  std::ranges::sort(points_, [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });

  auto uniq_range = std::ranges::unique(points_);  // возвращает subrange
  points_.erase(uniq_range.begin(), uniq_range.end());

  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }

  Pt pivot = points_[0];
  for (const Pt& p : points_) {
    if (p.second < pivot.second || (p.second == pivot.second && p.first < pivot.first)) {
      pivot = p;
    }
  }

  std::ranges::sort(points_, [&](const Pt& a, const Pt& b) {
    if (a == b) {
      return false;
    }
    long long cr = Cross(pivot, a, b);
    if (cr != 0) {
      return cr > 0;
    }
    return Dist2(pivot, a) < Dist2(pivot, b);
  });

  // Build hull (Graham scan)
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

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::PostProcessingImpl() {
  std::ranges::sort(output_, [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });

  Pt* dest = reinterpret_cast<Pt*>(task_data->outputs[0]);
  std::copy_n(output_.begin(), output_.size(), dest);
  task_data->outputs_count[0] = static_cast<int>(output_.size());
  return true;
}