from __future__ import annotations

import os
import time
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _rank_id() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else int(
            os.environ.get("RANK", "0")
        )
    except Exception:
        try:
            return int(os.environ.get("RANK", "0"))
        except ValueError:
            return 0


def _desc(name, value) -> str:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        text = (
            f"{name}=shape={tuple(value.shape)} dtype={value.dtype} "
            f"device={value.device} contiguous={value.is_contiguous()} "
            f"stride={tuple(value.stride())} storage_offset={value.storage_offset()}"
        )
        if value.numel() and name in {
            "actual_seq_lengths_query",
            "actual_seq_lengths_key",
            "actual_seq_lengths_kv",
            "block_table",
            "sparse_indices",
        }:
            try:
                head = value.flatten()[:16].detach().cpu().tolist()
                text += f" head={head}"
            except Exception as exc:
                text += f" head_error={exc!r}"
        return text
    return f"{name}={value!r}"


def _output_desc(value) -> str:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return _desc("out", value)
    if isinstance(value, tuple):
        return ", ".join(_output_desc(item) for item in value)
    return repr(value)


def _op_short_name(packet) -> str:
    name = getattr(packet, "_qualified_op_name", None)
    if not name:
        name = str(packet)
    return str(name).split("::")[-1].split(".")[-1]


def _maybe_dump(op_name: str, call_index: int, kwargs: dict) -> None:
    dump_dir = os.environ.get("SFA_TRACE_DUMP_DIR")
    if not dump_dir:
        return
    max_dumps = _env_int("SFA_TRACE_DUMP_MAX_CALLS", 1)
    if call_index > max_dumps:
        return
    try:
        import torch

        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "op": op_name,
            "rank": _rank_id(),
            "pid": os.getpid(),
            "call_index": call_index,
        }
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().cpu()
            else:
                payload[key] = value
        file_path = (
            path
            / f"{op_name}_rank{_rank_id()}_pid{os.getpid()}_"
            f"call{call_index:03d}.pt"
        )
        torch.save(payload, file_path)
        print(f"SFA_TRACE dumped {file_path}", flush=True)
    except Exception as exc:
        print(f"SFA_TRACE dump failed op={op_name} error={exc!r}", flush=True)


def install() -> None:
    if not _env_flag("SFA_TRACE_ENABLE", False):
        return

    import torch
    from torch._ops import OpOverloadPacket

    if getattr(OpOverloadPacket, "_sfa_trace_installed", False):
        return

    original_call = OpOverloadPacket.__call__
    targets = {
        "npu_lightning_indexer",
        "npu_lightning_indexer_quant",
        "npu_sparse_flash_attention",
    }
    counters: dict[str, int] = {}

    def traced_call(self, *args, **kwargs):
        op_name = _op_short_name(self)
        if op_name not in targets:
            return original_call(self, *args, **kwargs)

        counters[op_name] = counters.get(op_name, 0) + 1
        call_index = counters[op_name]
        max_calls = _env_int("SFA_TRACE_MAX_CALLS", 8)
        should_log = call_index <= max_calls
        rank = _rank_id()
        if should_log:
            print(
                "SFA_TRACE before "
                f"op={op_name} call={call_index} rank={rank} "
                f"pid={os.getpid()}",
                flush=True,
            )
            for index, value in enumerate(args):
                print(f"SFA_TRACE   {_desc(f'arg{index}', value)}", flush=True)
            for key in sorted(kwargs):
                print(f"SFA_TRACE   {_desc(key, kwargs[key])}", flush=True)
            _maybe_dump(op_name, call_index, kwargs)

        start = time.perf_counter()
        result = original_call(self, *args, **kwargs)
        if _env_flag("SFA_TRACE_SYNC", True):
            try:
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                print(
                    "SFA_TRACE sync failed "
                    f"op={op_name} call={call_index} rank={rank} "
                    f"error={exc!r}",
                    flush=True,
                )
                raise
        if should_log:
            print(
                "SFA_TRACE after "
                f"op={op_name} call={call_index} rank={rank} "
                f"pid={os.getpid()} elapsed={time.perf_counter() - start:.6f}s "
                f"{_output_desc(result)}",
                flush=True,
            )
        return result

    OpOverloadPacket.__call__ = traced_call
    OpOverloadPacket._sfa_trace_installed = True
    print(f"SFA_TRACE installed pid={os.getpid()} rank={_rank_id()}", flush=True)


install()
