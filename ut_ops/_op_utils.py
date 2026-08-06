from __future__ import annotations

import os
from pathlib import Path


def require_local_opapi() -> str:
    path = os.environ.get("NANOVLLM_CUST_OPAPI_LIB", "")
    if not path or not Path(path).is_file():
        raise RuntimeError(
            "Repository-local libcust_opapi.so was not selected; rebuild "
            "with `bash scripts/build_nanovllm_ops.sh`."
        )
    return path
