#!/usr/bin/env python3
"""Graph test: BF16 fused_li_manage -> scatter -> sparse-tail Attention."""

from _graph_bf16 import run


if __name__ == "__main__":
    run("split", __doc__ or "BF16 split offload graph test")
