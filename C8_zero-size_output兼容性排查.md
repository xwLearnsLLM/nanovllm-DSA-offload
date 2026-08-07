# C8 zero-size output descriptor 兼容性排查

## 结论

“wrapper 与实际加载的 ACLNN runtime 契约不匹配”很可能是正确方向，但笼统地归因于本仓库自定义算子的 C++ wrapper 并不准确。

当前仓库中，由我们维护的三个 C++ wrapper 没有在有效路径上传递 zero-size 输出：

- BF16 SFA 的 `softmax_max/softmax_sum` 使用 `{1}` 占位。
- LIDU 和 SCATTER 的输出都是固定非零形状。
- `op_api_common.cpp` 中虽然保留了一个构造 `empty({0})` 的 helper，但当前没有调用者。

C8 路径还会调用 vLLM-Ascend/torch_npu 提供的官方算子。zero-size output descriptor 更可能来自这些外部 wrapper。

## 两个可疑位置

### 官方 C8 Quant LightningIndexer

当 `return_value=False` 时，官方 wrapper 会为未使用的第二个输出创建 zero-size tensor：

```cpp
sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
```

本仓库的 C8 LIDU 会调用这个官方算子，并且只使用 Top2048 indices，不请求 sparse values。

### 官方 C8 QSFA

当 `return_softmax_lse=False` 时，官方 wrapper 会创建两个 zero-size 输出：

```cpp
softmax_size = {0};
softmax_max = at::empty(softmax_size, query.options().dtype(at::kFloat));
softmax_sum = at::empty(softmax_size, query.options().dtype(at::kFloat));
```

因此，如果问题出现在 C8 路径，zero-size descriptor 很可能来自官方 C8 LightningIndexer 或 QSFA wrapper，而不是本仓库的 LIDU cache-update、SCATTER kernel。

## 可能原因

按可能性从高到低排列：

1. `torch_npu`、op-plugin 或 vLLM-Ascend 与实际加载的 CANN runtime 不配套。
2. 机器安装了多套 CANN：编译使用一套头文件，运行时通过 `LD_LIBRARY_PATH` 加载了另一套动态库。
3. 两台机器走了不同的 Python dispatch：
   - C8 LI 可能走 `torch_npu.npu_quant_lightning_indexer`；
   - 也可能走 `torch.ops._C_ascend.npu_lightning_indexer_quant`；
   - C8 QSFA 同样可能来自 `torch_npu` 或 `_C_ascend`。
4. 使用了旧的 `_C_ascend.so`、旧 torch extension 或旧自定义 OPP。
5. 单纯的 AICore kernel bug 可能性较低。zero-size descriptor 报错一般发生在 host 参数校验阶段，此时 kernel 尚未启动。

PyTorch 本身允许 zero-size tensor；存在兼容性差异的是特定 ACLNN 算子、wrapper 和 runtime 对 zero-size output descriptor 的约定。

## 根据报错中的算子名定位

- `aclnnQuantLightningIndexer`：重点检查未使用的 `sparse_values` 输出。
- `aclnnKvQuantSparseFlashAttention`：重点检查未使用的 `softmax_max/softmax_sum` 输出。
- `aclnnA5SparseAndTailAttention`：确认是否使用了旧二进制；当前仓库的 BF16 wrapper 已使用 `{1}` 占位，不会主动创建 zero-size softmax 输出。
- LIDU cache-update 或 SCATTER：这两个自定义算子本身没有 zero-size 输出，需要检查实际加载的 `.so` 是否来自当前代码。

请保留完整异常栈，尤其是 `call aclnnXXX failed`、输入输出 descriptor 和实际加载动态库路径。

## 两台机器的环境对比

在已经通过测试的机器和报错机器上分别执行以下命令，然后逐项对比输出：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

readlink -f "$ASCEND_HOME_PATH"
which python3
python3 -m pip show torch torch-npu vllm vllm-ascend

python3 - <<'PY'
import os
import torch
import torch_npu

ns = getattr(torch.ops, "_C_ascend", None)

print("torch                  =", torch.__version__)
print("torch_npu              =", torch_npu.__version__)
print("torch_npu path         =", torch_npu.__file__)
print("ASCEND_HOME_PATH       =", os.getenv("ASCEND_HOME_PATH"))
print("CANN_INSTALL_PATH      =", os.getenv("CANN_INSTALL_PATH"))
print("LD_LIBRARY_PATH        =", os.getenv("LD_LIBRARY_PATH"))
print("ASCEND_CUSTOM_OPP_PATH =", os.getenv("ASCEND_CUSTOM_OPP_PATH"))

print(
    "torch_npu C8 LI       =",
    getattr(torch_npu, "npu_quant_lightning_indexer", None) is not None,
)
print(
    "_C_ascend C8 LI       =",
    ns is not None
    and getattr(ns, "npu_lightning_indexer_quant", None) is not None,
)
print(
    "torch_npu C8 QSFA     =",
    getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None) is not None,
)
print(
    "_C_ascend C8 QSFA     =",
    ns is not None
    and getattr(ns, "npu_kv_quant_sparse_flash_attention", None) is not None,
)

torch.npu.set_device(0)
x = torch.empty((0,), dtype=torch.float32, device="npu")
print("plain zero tensor      =", x.shape, x.numel(), x.data_ptr())

loaded = set()
for line in open("/proc/self/maps"):
    if any(
        name in line
        for name in ("libascendcl", "libnnopbase", "libopapi", "libtorch_npu")
    ):
        loaded.add(line.split()[-1])

print("loaded runtime libs:")
for path in sorted(loaded):
    print(" ", path)
PY
```

仅确认“都使用 CANN 9.1”不够，还需要对齐：

- CANN 的完整补丁版本，例如 `9.1.T560`；
- `torch` 和 `torch_npu` 版本；
- op-plugin/ops-transformer 版本；
- vLLM-Ascend 版本或 commit；
- 实际 `_C_ascend.so` 来源；
- `ASCEND_HOME_PATH` 的真实路径；
- `LD_LIBRARY_PATH` 中 CANN 动态库的优先顺序；
- 实际加载的 `libascendcl.so`、`libnnopbase.so` 和 op-api 动态库。

## 不建议直接修改 `{0}` 为 `{1}`

不要只在官方 wrapper 中把 `{0}` 改成 `{1}`。官方 op 的 InferShape 在相应 flag 为 false 时也可能推导 `{0}`；只修改 wrapper 会导致输出 tensor descriptor 与算子 schema/InferShape 再次不一致。

可以用以下方法做临时穿刺诊断，但不建议直接作为生产修复：

- Quant LightningIndexer 设置 `return_value=True`。如果错误消失，基本可以确认问题来自 zero-size `sparse_values`。
- QSFA 设置 `return_softmax_lse=True`。如果错误消失，基本可以确认问题来自 zero-size `softmax_max/softmax_sum`。

这两个选项可能增加计算量、输出内存和图内开销。生产修复应优先让 wrapper、op definition、torch_npu/op-plugin 与 ACLNN runtime 使用同一套已验证的软件版本。
