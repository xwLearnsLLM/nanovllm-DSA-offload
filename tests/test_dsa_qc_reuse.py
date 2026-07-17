from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nanovllm.models import dsa_indexer_project as project


def test_eager_dispatch_gather_preserves_tensor_dependency(monkeypatch):
    inputs = tuple(torch.tensor([index]) for index in range(10))
    observed = {}

    def fake_dispatch(*args):
        observed["args"] = args
        return args[0], args[1], args[2], args[3]

    monkeypatch.setattr(
        project,
        "_GRAPH_GATHER_SELECTION_KV_CACHE",
        fake_dispatch,
    )

    outputs = project.gather_selection_kv_cache_eager_dispatch(*inputs)

    assert observed["args"] == inputs
    assert outputs == inputs[:4]


def test_full_graph_pipeline_consumes_mlapo_q_c(monkeypatch):
    batch_size = 2
    n_head = 1
    head_dim = 4
    rope_dim = 2
    sparse_count = 2

    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    q_c = torch.tensor(
        [[0.5, -1.0, 2.0], [3.0, 0.25, -0.5]],
    )
    wq_b_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    weights_proj_weight = torch.ones(n_head, hidden_states.shape[-1])
    cos = torch.ones(batch_size, rope_dim)
    sin = torch.zeros(batch_size, rope_dim)

    observed = {}

    def fake_lightning_indexer(**kwargs):
        observed["query"] = kwargs["query"].clone()
        return torch.zeros(
            batch_size,
            1,
            1,
            sparse_count,
            dtype=torch.int32,
        )

    def fake_gather_selection(*args):
        observed["gather_called"] = True
        observed["gather_full_kv_lens"] = args[-1].clone()

    monkeypatch.setattr(project, "_GRAPH_LIGHTNING_INDEXER", None)
    monkeypatch.setattr(project, "_GRAPH_GATHER_SELECTION_KV_CACHE", None)
    monkeypatch.setattr(
        project,
        "ascend_ops",
        SimpleNamespace(
            npu_lightning_indexer=fake_lightning_indexer,
            npu_gather_selection_kv_cache=fake_gather_selection,
        ),
    )

    q_index, _, _ = project._dsa_indexer_pipeline_with_qc_functional(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        torch.empty(batch_size, n_head, head_dim),
        torch.empty(batch_size, n_head),
        torch.empty(1),
        torch.ones(batch_size, dtype=torch.int32),
        torch.full((batch_size,), 8, dtype=torch.int32),
        torch.zeros(batch_size, 1, dtype=torch.int32),
        torch.empty(1),
        torch.empty(1),
        torch.zeros(batch_size, 1, dtype=torch.int32),
        torch.empty(1),
        torch.arange(batch_size, dtype=torch.int32),
        torch.empty(1),
        torch.empty(1),
        torch.zeros(batch_size, 1, dtype=torch.int32),
        n_head=n_head,
        head_dim=head_dim,
        rope_dim=rope_dim,
        score_scale=1.0,
        sparse_count=sparse_count,
    )

    expected = F.linear(q_c, wq_b_weight).view(
        batch_size,
        n_head,
        head_dim,
    )
    assert torch.equal(q_index, expected)
    assert torch.equal(observed["query"], expected)
    assert observed["gather_called"] is True
    # GatherSelection excludes the current query internally, so its full-KV
    # length must be one larger than the prefix-only LightningIndexer length.
    assert observed["gather_full_kv_lens"].tolist() == [9, 9]
