// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "common/QueryResult.h"
#include "common/Types.h"
#include "segcore/Utils.h"

namespace milvus::segcore {

// Regression for #54381: distinct scores within EPSILON must follow their PKs.
TEST(SortEqualScoresByPks, PreservesNearEqualDistances) {
    SearchResult result;
    result.total_nq_ = 1;
    result.unity_topK_ = 2;
    result.pk_type_ = DataType::INT64;
    // L2 scores are negated before export, with larger scores first.
    result.distances_ = {0.0f, -1e-10f};
    result.seg_offsets_ = {20, 10};
    result.primary_keys_ = {PkType(int64_t(2)), PkType(int64_t(1))};
    result.topk_per_nq_prefix_sum_ = {0, 2};

    SortEqualScoresByPks(&result);

    EXPECT_EQ(std::get<int64_t>(result.primary_keys_[0]), 1);
    EXPECT_EQ(std::get<int64_t>(result.primary_keys_[1]), 2);
    EXPECT_EQ(result.seg_offsets_, (std::vector<int64_t>{10, 20}));
    EXPECT_EQ(result.distances_, (std::vector<float>{-1e-10f, 0.0f}));
}

TEST(SortEqualScoresByPks, PreservesStringPkDistances) {
    SearchResult result;
    result.total_nq_ = 1;
    result.unity_topK_ = 2;
    result.pk_type_ = DataType::VARCHAR;
    result.distances_ = {2e-10f, 1e-10f};
    result.seg_offsets_ = {20, 10};
    result.primary_keys_ = {PkType(std::string("b")), PkType(std::string("a"))};
    result.topk_per_nq_prefix_sum_ = {0, 2};

    SortEqualScoresByPks(&result);

    EXPECT_EQ(std::get<std::string>(result.primary_keys_[0]), "a");
    EXPECT_EQ(std::get<std::string>(result.primary_keys_[1]), "b");
    EXPECT_EQ(result.seg_offsets_, (std::vector<int64_t>{10, 20}));
    EXPECT_EQ(result.distances_, (std::vector<float>{1e-10f, 2e-10f}));
}

TEST(SortEqualScoresByPks, PreservesOptionalFieldsThroughLongCycle) {
    SearchResult result;
    result.total_nq_ = 1;
    result.unity_topK_ = 3;
    result.pk_type_ = DataType::INT64;
    result.element_level_ = true;
    result.distances_ = {3e-10f, 2e-10f, 1e-10f};
    result.seg_offsets_ = {30, 10, 20};
    result.primary_keys_ = {
        PkType(int64_t(3)), PkType(int64_t(1)), PkType(int64_t(2))};
    result.element_indices_ = {33, 11, 22};
    auto make_group_key = [](int64_t value) {
        CompositeGroupKey key;
        key.Add(GroupByValueType(value));
        return key;
    };
    result.composite_group_by_values_ = std::vector<CompositeGroupKey>{
        make_group_key(3), make_group_key(1), make_group_key(2)};
    result.topk_per_nq_prefix_sum_ = {0, 3};

    SortEqualScoresByPks(&result);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(std::get<int64_t>(result.primary_keys_[i]),
                  static_cast<int64_t>(i + 1));
        EXPECT_EQ(std::get<int64_t>(
                      result.composite_group_by_values_.value()[i][0].value()),
                  static_cast<int64_t>(i + 1));
    }
    EXPECT_EQ(result.seg_offsets_, (std::vector<int64_t>{10, 20, 30}));
    EXPECT_EQ(result.element_indices_, (std::vector<int32_t>{11, 22, 33}));
    EXPECT_EQ(result.distances_,
              (std::vector<float>{2e-10f, 1e-10f, 3e-10f}));
}

TEST(SortEqualScoresByPks, PreservesQueryAndScoreGroupBoundaries) {
    SearchResult result;
    result.total_nq_ = 4;
    result.unity_topK_ = 6;
    result.pk_type_ = DataType::INT64;
    // Empty query, singleton, mixed score groups, then another near-equal run.
    result.topk_per_nq_prefix_sum_ = {0, 0, 1, 7, 9};
    result.distances_ = {
        0.0f, 3.0f, 0.0f, -1e-10f, -2e-10f, -1.0f, -1.0f, 0.0f, -3e-10f};
    result.seg_offsets_ = {90, 80, 30, 10, 20, 60, 50, 40, 10};
    result.primary_keys_ = {PkType(int64_t(9)),
                            PkType(int64_t(8)),
                            PkType(int64_t(3)),
                            PkType(int64_t(1)),
                            PkType(int64_t(2)),
                            PkType(int64_t(6)),
                            PkType(int64_t(5)),
                            PkType(int64_t(4)),
                            PkType(int64_t(1))};

    SortEqualScoresByPks(&result);

    const std::vector<int64_t> expected_pks = {9, 8, 1, 2, 3, 5, 6, 1, 4};
    for (size_t i = 0; i < expected_pks.size(); ++i) {
        EXPECT_EQ(std::get<int64_t>(result.primary_keys_[i]), expected_pks[i]);
    }
    EXPECT_EQ(result.seg_offsets_,
              (std::vector<int64_t>{90, 80, 10, 20, 30, 50, 60, 10, 40}));
    EXPECT_EQ(result.distances_,
              (std::vector<float>{0.0f,
                                  3.0f,
                                  -1e-10f,
                                  -2e-10f,
                                  0.0f,
                                  -1.0f,
                                  -1.0f,
                                  -3e-10f,
                                  0.0f}));
    EXPECT_EQ(result.topk_per_nq_prefix_sum_,
              (std::vector<size_t>{0, 0, 1, 7, 9}));
}

}  // namespace milvus::segcore
