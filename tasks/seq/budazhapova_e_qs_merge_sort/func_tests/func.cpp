#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/budazhapova_e_qs_merge_sort/include/inc.h"


TEST(budazhapova_e_qs_merge_sort_seq, test_sort_small_array) {
	std::vector<int> in = { 5, 2, 8, 1, 9, 3, 7, 4, 6, 0 };
	std::vector<int> expected = in;
	std::sort(expected.begin(), expected.end());
	std::vector<int> out(in.size(), 0);

	// Create task_data
	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	// Create Task
	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_already_sorted) {
	std::vector<int> in = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	std::vector<int> expected = in;
	std::vector<int> out(in.size(), 0);

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_reverse_sorted) {
	std::vector<int> in = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
	std::vector<int> expected = in;
	std::sort(expected.begin(), expected.end());
	std::vector<int> out(in.size(), 0);

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_with_duplicates) {
	std::vector<int> in = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
	std::vector<int> expected = in;
	std::sort(expected.begin(), expected.end());
	std::vector<int> out(in.size(), 0);

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_single_element) {
	std::vector<int> in = { 42 };
	std::vector<int> expected = in;
	std::vector<int> out(in.size(), 0);

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_empty_array) {
	std::vector<int> in = {};
	std::vector<int> expected = in;
	std::vector<int> out(in.size(), 0);

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_seq, test_sort_large_random_array) {
	constexpr size_t kCount = 1000;
	std::vector<int> in(kCount);
	std::vector<int> out(kCount, 0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(-1000, 1000);

	for (size_t i = 0; i < kCount; ++i) {
		in[i] = dist(gen);
	}

	std::vector<int> expected = in;
	std::sort(expected.begin(), expected.end());

	auto task_data_seq = std::make_shared<ppc::core::TaskData>();
	task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	task_data_seq->inputs_count.emplace_back(in.size());
	task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	task_data_seq->outputs_count.emplace_back(out.size());

	budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential test_task_sequential(task_data_seq);
	ASSERT_EQ(test_task_sequential.Validation(), true);
	test_task_sequential.PreProcessing();
	test_task_sequential.Run();
	test_task_sequential.PostProcessing();

	EXPECT_EQ(out, expected);
}