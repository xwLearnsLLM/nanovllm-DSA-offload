#!/usr/bin/env python3
"""Graph test: BF16 fused_li_manage -> fused copy+sparse-tail Attention."""

from _graph_bf16 import run


if __name__ == "__main__":
    run("fused", __doc__ or "BF16 fused offload graph test")
