/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QK_SCORE_TORCH_ADPT_H
#define QK_SCORE_TORCH_ADPT_H
namespace vllm_ascend {

inline int64_t get_qk_score_s2_size(
    const at::Tensor &key,
    const c10::optional<at::Tensor> &block_table,
    const std::string &key_layout_str)
{
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    if (key_layout_str == "PA_BSND") {
        TORCH_CHECK(block_table.has_value(), "block_table must be provided when layout_key='PA_BSND'.");
        TORCH_CHECK(key.dim() == 4, "key must be 4-D when layout_key='PA_BSND'.");
        TORCH_CHECK(block_table.value().dim() == 2, "block_table must be 2-D.");
        return block_table.value().size(DIM_1) * key.size(DIM_1);
    }
    if (key_layout_str == "TND") {
        return key.size(DIM_0);
    }
    return key.size(DIM_1);
}

inline int64_t get_qk_score_kv_heads(const at::Tensor &key, const std::string &key_layout_str)
{
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    return key_layout_str == "TND" ? key.size(DIM_1) : key.size(DIM_2);
}

at::Tensor npu_qk_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key)
{
    // npu tensor max size
    constexpr int32_t SIZE = 8;
    constexpr int32_t DIM_0 = 0;
    constexpr int32_t DIM_1 = 1;
    TORCH_CHECK(query.numel() > 0, "Query is empty.");
    TORCH_CHECK(key.numel() > 0, "Key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Weights is empty.");
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    int64_t score_count = get_qk_score_s2_size(key, block_table, key_layout_str);
    int64_t kv_heads = get_qk_score_kv_heads(key, key_layout_str);
    TORCH_CHECK(score_count > 0, "qk score output length should be greater than 0, but now is ", score_count);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), kv_heads, score_count};
    } else {
        output_size = {query.size(DIM_0), kv_heads, score_count};
    }
    at::Tensor qk_score_output = at::empty(output_size, query.options().dtype(at::kFloat));
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    EXEC_NPU_CMD(
        aclnnQkScore,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        query_layout_ptr,
        key_layout_ptr,
        score_count,
        qk_score_output);
    return qk_score_output;
}
}
#endif
