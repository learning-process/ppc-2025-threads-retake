#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/matyunina_a_constructing_convex_hull/include/ops_seq.hpp"

namespace {
void RunTest(int height, int width, std::vector<int> image, const
             std::vector<matyunina_a_constructing_convex_hull_seq::Point>& ans) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);

  matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull constructing_convex_hull(task_data);
  ASSERT_TRUE(constructing_convex_hull.Validation());
  constructing_convex_hull.PreProcessing();
  constructing_convex_hull.Run();
  constructing_convex_hull.PostProcessing();

  auto point_array =
      reinterpret_cast<matyunina_a_constructing_convex_hull_seq::Point*>(task_data->outputs[0]);
  std::vector<matyunina_a_constructing_convex_hull_seq::Point> points(point_array,
                                                                      point_array + task_data->outputs_count[0]);

  // std::cout << "\n#######\n";
  // for (matyunina_a_constructing_convex_hull_seq::Point& point: points) {
  //   std::cout<< "x: " << point.x << " y: " << point.y << "\n";
  // }
  // std::cout << "\n#######\n";

  EXPECT_EQ(points, ans);
}

#ifndef _WIN32
bool RunImageTest(const std::string& image_path, const std::string& ans_path) {
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  cv::threshold(image, image, 128, 1, cv::THRESH_BINARY_INV);

  std::vector<int> pixels(image.rows * image.cols);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      pixels[((i * image.cols) + j)] = image.at<uchar>(i, j);
    }
  }

  int width = image.cols;
  int height = image.rows;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);

  matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull constructing_convex_hull(task_data);
  constructing_convex_hull.Validation();
  constructing_convex_hull.PreProcessing();
  constructing_convex_hull.Run();
  constructing_convex_hull.PostProcessing();

  auto point_array =
      reinterpret_cast<matyunina_a_constructing_convex_hull_seq::Point*>(task_data->outputs[0]);
  std::vector<matyunina_a_constructing_convex_hull_seq::Point> points(point_array,
                                                                      point_array + task_data->outputs_count[0]);

  // std::cout << "\n#######\n";
  // for (matyunina_a_constructing_convex_hull_seq::Point& point : points) {
  //   std::cout << "x: " << point.x << " y: " << point.y << "\n";
  // }
  // std::cout << "\n#######\n";

  cv::Mat ans = cv::imread(ans_path);
  if (ans.empty()) {
    return false;
  }
  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans_points;

  for (int i = 0; i < ans.cols; i++) {
    for (int j = 0; j < ans.rows; j++) {
      cv::Vec3b pixel = ans.at<cv::Vec3b>(j, i);
      uchar blue = pixel[0];
      uchar green = pixel[1];
      uchar red = pixel[2];

      if (red > 200 && green < 50 && blue < 50) {
        ans_pointsemplace_back(i, j);
      }
    }
  }

  image.release();
  ans.release();

  return ans_points == points;
}
#endif
}  // namespace

#ifndef _WIN32
TEST(matyunina_a_constructing_convex_hull_seq, image1) {
  std::string image_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/image1.png");
  std::string ans_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/ans1.png");

  ASSERT_TRUE(RunImageTest(image_path, ans_path));
}
TEST(matyunina_a_constructing_convex_hull_seq, image2) {
  std::string image_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/image2.png");
  std::string ans_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/ans2.png");

  ASSERT_TRUE(RunImageTest(image_path, ans_path));
}
TEST(matyunina_a_constructing_convex_hull_seq, image3) {
  std::string image_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/image3.png");
  std::string ans_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/ans3.png");

  ASSERT_TRUE(RunImageTest(image_path, ans_path));
}
TEST(matyunina_a_constructing_convex_hull_seq, image4) {
  std::string image_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/image4.png");
  std::string ans_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/ans4.png");

  ASSERT_TRUE(RunImageTest(image_path, ans_path));
}
TEST(matyunina_a_constructing_convex_hull_seq, image5) {
  std::string image_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/image5.png");
  std::string ans_path = ppc::util::GetAbsolutePath("seq/matyunina_a_constructing_convex_hull/data/ans5.png");

  ASSERT_TRUE(RunImageTest(image_path, ans_path));
}
#endif

TEST(matyunina_a_constructing_convex_hull_seq, test_9_9) {
  int height = 9;
  int width = 9;
  // clang-format off
  std::vector<int> image = {
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {1, 3},
    {1, 5},
    {3, 1},
    {3, 7},
    {5, 1},
    {5, 7},
    {7, 3},
    {7, 5}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_4_4_square) {
  int height = 4;
  int width = 4;
  // clang-format off
  std::vector<int> image = {
    0,0,0,0,
    0,1,1,0,
    0,1,1,0,
    0,0,0,0
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {1, 1},
    {1, 2},
    {2, 1},
    {2, 2}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_4_4_polygon) {
  int height = 4;
  int width = 4;
  // clang-format off
  std::vector<int> image = {
    0,0,0,1,
    0,1,1,0,
    0,1,1,0,
    0,0,0,0
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {1, 1},
    {1, 2},
    {2, 2},
    {3, 0}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_6_3_line) {
  int height = 3;
  int width = 6;
  // clang-format off
  std::vector<int> image = {
    0,0,0,0,0,0,
    0,1,1,1,1,0,
    0,0,0,0,0,0
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {1, 1},
    {4, 1},
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_empty_square) {
  int height = 4;
  int width = 4;
  // clang-format off
  std::vector<int> image = {
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans;
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_black_square) {
  int height = 4;
  int width = 4;
  // clang-format off
  std::vector<int> image = {
    1,1,1,1,
    1,1,1,1,
    1,1,1,1,
    1,1,1,1
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {0, 0},
    {0, 3},
    {3, 0},
    {3, 3}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_triangle) {
  int height = 4;
  int width = 5;
  // clang-format off
  std::vector<int> image = {
    0,0,1,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    1,0,0,0,1
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {0, 3},
    {2, 0},
    {4, 3}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_9_9_polygon) {
  int height = 9;
  int width = 9;
  // clang-format off
  std::vector<int> image = {
    0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0,
    0, 1, 0, 1, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {0, 8},
    {1, 3},
    {3, 1},
    {5, 0},
    {5, 7},
    {7, 1},
    {8, 6}
  };
  // clang-format on

  RunTest(height, width, image, ans);
}