#ifndef OPS_OP_PROTO_INC_DSA_INDEX_UPDATE_H_
#define OPS_OP_PROTO_INC_DSA_INDEX_UPDATE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(DsaIndexUpdate)
    .INPUT(score, TensorType({DT_BF16}))
    .INPUT(hbm_cached_tokens_pool, TensorType({DT_INT32}))
    .INPUT(candidate_lens, TensorType({DT_INT32}))
    .INPUT(selected_lens, TensorType({DT_INT32}))
    .INPUT(req_pool_entries, TensorType({DT_INT32}))
    .OUTPUT(promote_idx, TensorType({DT_INT32}))
    .OUTPUT(demote_idx, TensorType({DT_INT32}))
    .OUTPUT(copy_counts, TensorType({DT_INT32}))
    .REQUIRED_ATTR(max_copy_tokens, Int)
    .OP_END_FACTORY_REG(DsaIndexUpdate)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DSA_INDEX_UPDATE_H_
