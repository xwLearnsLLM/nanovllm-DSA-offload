# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from __future__ import annotations

import ctypes
from typing import Union
import torch
import numpy as np

"""
类型预留
"""

# 使用字符串形式避免循环导入，from __future__ import annotations 会自动处理
SupportedTensor = Union[torch.Tensor, np.ndarray, "OpTensor"]  # type: ignore[name-defined]
SupportedDataType = Union[torch.dtype, np.dtype]


class GM_ADDR(ctypes.c_void_p):
    pass

# 未实现
class EpilogueParams(ctypes.c_void_p):
    pass
