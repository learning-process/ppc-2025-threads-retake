#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/matyunina_a_constructing_convex_hull/include/ops_seq.hpp"

namespace {
  void RunTest(int height, int width, std::vector<int> image, std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans) {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data->inputs_count.emplace_back(width);
    task_data->inputs_count.emplace_back(height);

    matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull constructingConvexHull(task_data);
    ASSERT_TRUE(constructingConvexHull.Validation());
    constructingConvexHull.PreProcessing();
    constructingConvexHull.Run();
    constructingConvexHull.PostProcessing();

    matyunina_a_constructing_convex_hull_seq::Point* pointArray = reinterpret_cast<matyunina_a_constructing_convex_hull_seq::Point*>(task_data->outputs[0]);
    std::vector<matyunina_a_constructing_convex_hull_seq::Point> points(pointArray, pointArray + task_data->outputs_count[0]);

    // std::cout << "\n#######\n";
    // for (matyunina_a_constructing_convex_hull_seq::Point& point: points) {
    //   std::cout<< "x: " << point.x << " y: " << point.y << "\n";
    // }
    // std::cout << "\n#######\n";

    EXPECT_EQ(points, ans);
  }
}

TEST(matyunina_a_constructing_convex_hull_seq, test_9_9) {
  int height = 9;
  int width = 9;

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

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_4_4_square) {
  int height = 4;
  int width = 4;

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

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_4_4_polygon) {
  int height = 4;
  int width = 4;

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

  RunTest(height, width, image, ans);
}

TEST(matyunina_a_constructing_convex_hull_seq, test_6_3_line) {
  int height = 3;
  int width = 6;

  std::vector<int> image = {
    0,0,0,0,0,0,
    0,1,1,1,1,0,
    0,0,0,0,0,0
  };

  std::vector<matyunina_a_constructing_convex_hull_seq::Point> ans = {
    {1, 1},
    {4, 1},
   };

  RunTest(height, width, image, ans);
}