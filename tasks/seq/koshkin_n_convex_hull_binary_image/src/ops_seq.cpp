#include "seq/koshkin_n_convex_hull_binary_image/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stack>
#include <utility>
#include <vector>

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::PreProcessingImpl() {
  width = static_cast<int>(task_data->inputs_count[0]);
  height = static_cast<int>(task_data->inputs_count[1]);
  unsigned int size = width * height;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input = std::vector<int>(in_ptr, in_ptr + size);

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
  return bax * cay - bay * cax;
}

long long koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::Dist2(const Pt& a, const Pt& b) {
  long long dx = static_cast<long long>(a.first) - b.first;
  long long dy = static_cast<long long>(a.second) - b.second;
  return dx * dx + dy * dy;
}

void koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::FindPoints() {
  points.clear();
  if (width <= 0 || height <= 0) return;

  for (int j = 0; j < height; ++j) {
    int row_off = j * width;
    for (int i = 0; i < width; ++i) {
      if (input[row_off + i] != 1) continue;
      bool is_border = false;
      for (int dj = -1; dj <= 1 && !is_border; ++dj) {
        int nj = j + dj;
        if (nj < 0 || nj >= height) {
          is_border = true;
          break;
        }
        int noff = nj * width;
        for (int di = -1; di <= 1; ++di) {
          if (di == 0 && dj == 0) continue;
          int ni = i + di;
          if (ni < 0 || ni >= width) {
            is_border = true;
            break;
          }
          if (input[noff + ni] == 0) {
            is_border = true;
            break;
          }
        }
      }
      if (is_border) points.emplace_back(i, j);
    }
  }
}

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::RunImpl() {
  FindPoints();

  if (points.empty()) {
    output.clear();
    return true;
  }

  std::ranges::sort(points, [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());

  if (points.size() < 3) {
    output = points;
    return true;
  }

  Pt pivot = points[0];
  for (const Pt& p : points) {
    if (p.second < pivot.second || (p.second == pivot.second && p.first < pivot.first)) {
      pivot = p;
    }
  }

  std::ranges::sort(points, [&](const Pt& a, const Pt& b) {
    if (a == b) return false;
    long long cr = Cross(pivot, a, b);
    if (cr != 0) return cr > 0;
    return Dist2(pivot, a) < Dist2(pivot, b);
  });

  // Build hull (Graham scan)
  std::vector<Pt> hull;
  hull.reserve(points.size());
  for (const Pt& p : points) {
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

  if (hull.size() == 1 && points.size() > 1) {
    hull.push_back(points.back());
  }

  output = std::move(hull);
  return true;
}

bool koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage::PostProcessingImpl() {
  std::ranges::sort(output, [](const Pt& a, const Pt& b) {
    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
  });

  Pt* dest = reinterpret_cast<Pt*>(task_data->outputs[0]);
  std::copy_n(output.begin(), output.size(), dest);
  task_data->outputs_count[0] = static_cast<int>(output.size());
  return true;
}