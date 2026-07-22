from __future__ import annotations

import torch


def shifted_mtp_prefill_tokens(
    token_ids: list[int],
    start: int,
    end: int,
    *,
    sampled_token_id: int | None = None,
) -> list[int]:
    """Pair target hidden[p] with token[p + 1] for one prefill chunk.

    The final prompt position has no prompt-side successor, so its shifted
    input is the first token sampled by the target model.
    """

    if not 0 <= start < end <= len(token_ids):
        raise ValueError(
            "Invalid MTP prefill range: "
            f"start={start}, end={end}, length={len(token_ids)}."
        )
    if end < len(token_ids):
        if sampled_token_id is not None:
            raise ValueError(
                "An intermediate MTP prefill chunk must not have a sampled "
                "token."
            )
        return list(token_ids[start + 1 : end + 1])
    if sampled_token_id is None:
        raise ValueError(
            "The final MTP prefill chunk requires the target sampled token."
        )
    return list(token_ids[start + 1 : end]) + [int(sampled_token_id)]


def greedy_prefix_accept(
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accepted draft counts and the mismatch/bonus target token."""

    if target_token_ids.dim() != 2 or draft_token_ids.dim() != 2:
        raise ValueError("Greedy MTP verification expects rank-2 tensors.")
    if target_token_ids.shape[0] != draft_token_ids.shape[0]:
        raise ValueError("Target and draft batch sizes must match.")
    if target_token_ids.shape[1] != draft_token_ids.shape[1] + 1:
        raise ValueError(
            "Target verification must contain K+1 predictions for K drafts."
        )

    matches = target_token_ids[:, :-1].eq(draft_token_ids)
    accepted_counts = torch.cumprod(
        matches.to(torch.int32), dim=1
    ).sum(dim=1).to(torch.long)
    rows = torch.arange(
        target_token_ids.shape[0],
        dtype=torch.long,
        device=target_token_ids.device,
    )
    return accepted_counts, target_token_ids[rows, accepted_counts]


def materialize_accepted_tokens(
    draft_token_ids: list[list[int]],
    target_token_ids: list[list[int]],
    accepted_counts: list[int],
) -> list[list[int]]:
    if not (
        len(draft_token_ids)
        == len(target_token_ids)
        == len(accepted_counts)
    ):
        raise ValueError("Greedy MTP result batch sizes must match.")
    result: list[list[int]] = []
    for drafts, targets, count in zip(
        draft_token_ids, target_token_ids, accepted_counts
    ):
        if not 0 <= count <= len(drafts) or len(targets) != len(drafts) + 1:
            raise ValueError("Invalid greedy MTP acceptance row.")
        result.append(list(drafts[:count]) + [int(targets[count])])
    return result
