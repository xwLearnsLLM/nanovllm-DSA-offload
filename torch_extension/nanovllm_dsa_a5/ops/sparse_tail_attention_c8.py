from __future__ import annotations

import torch


# The schema, PrivateUse1 implementation and Meta implementation all live in
# the C++ extension. No Python custom-op or native vLLM-Ascend QSFA adapter is
# involved, so ACLGraph sees one repository-local device operator.
sparse_tail_attention_c8 = torch.ops.nanovllm_dsa.sparse_tail_attention_c8.default
