import torch
import torch.nn.functional as F

from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project,
    dsa_indexer_project_query_only,
)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.reshape(cos.shape[0], -1)
    sin = sin.reshape(sin.shape[0], -1)
    view = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[-1],)
    return x * cos.view(view) + _rotate_interleaved(x) * sin.view(view)


def test_glm_interleaved_indexer_full_and_query_only_match_golden():
    torch.manual_seed(7)
    tokens = 3
    hidden_size = 5
    q_lora_rank = 6
    n_head = 2
    head_dim = 4
    rope_dim = 4

    hidden_states = torch.randn(tokens, hidden_size)
    q_c = torch.randn(tokens, q_lora_rank)
    wq_b = torch.randn(n_head * head_dim, q_lora_rank)
    wk = torch.randn(head_dim, hidden_size)
    k_norm_weight = torch.randn(head_dim)
    k_norm_bias = torch.randn(head_dim)
    weights_proj = torch.randn(n_head, hidden_size)

    angles = torch.tensor(
        [[0.2, -0.4], [0.5, 0.8], [-0.7, 0.3]],
        dtype=torch.float32,
    )
    cos = angles.cos().repeat_interleave(2, dim=-1).view(tokens, 1, 1, rope_dim)
    sin = angles.sin().repeat_interleave(2, dim=-1).view(tokens, 1, 1, rope_dim)

    q_out = torch.empty(tokens, n_head, head_dim)
    k_out = torch.empty(tokens, head_dim)
    weights_out = torch.empty(tokens, n_head)
    dsa_indexer_project(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b,
        wk,
        k_norm_weight,
        k_norm_bias,
        weights_proj,
        q_out,
        k_out,
        weights_out,
        n_head=n_head,
        head_dim=head_dim,
        rope_dim=rope_dim,
        score_scale=1.0,
    )

    q = F.linear(q_c, wq_b).view(tokens, n_head, head_dim)
    k = F.layer_norm(
        F.linear(hidden_states, wk),
        (head_dim,),
        k_norm_weight,
        k_norm_bias,
        eps=1e-6,
    )
    q_expected = _apply_interleaved_rope(q, cos, sin)
    k_expected = _apply_interleaved_rope(k.unsqueeze(1), cos, sin).squeeze(1)
    weights_expected = F.linear(hidden_states, weights_proj)

    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)
    torch.testing.assert_close(weights_out, weights_expected)

    query_only_q = torch.empty_like(q_out)
    query_only_weights = torch.empty_like(weights_out)
    dsa_indexer_project_query_only(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b,
        weights_proj,
        query_only_q,
        query_only_weights,
        n_head=n_head,
        head_dim=head_dim,
        rope_dim=rope_dim,
        score_scale=1.0,
    )
    torch.testing.assert_close(query_only_q, q_expected)
    torch.testing.assert_close(query_only_weights, weights_expected)


def test_glm_interleaved_rope_differs_from_neox_for_nontrivial_angles():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    angles = torch.tensor([[0.2, 0.7]])
    interleaved_cos = angles.cos().repeat_interleave(2, dim=-1)
    interleaved_sin = angles.sin().repeat_interleave(2, dim=-1)
    interleaved = _apply_interleaved_rope(
        x,
        interleaved_cos,
        interleaved_sin,
    )
    neox_rotate = torch.cat((-x[..., 2:], x[..., :2]), dim=-1)
    neox = x * interleaved_cos.view(1, 1, 4) + neox_rotate * interleaved_sin.view(1, 1, 4)
    assert not torch.allclose(interleaved, neox)
