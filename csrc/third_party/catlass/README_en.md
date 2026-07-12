# CATLASS

---

## ⚠ Important Changes

At the first community meeting in March 2026, we officially confirmed that the CATLASS community mainline will add support for the next-generation Ascend hardware Ascend 950PR/Ascend 950DT. To distinguish the underlying interface implementations on different platforms, this new support will introduce a new compilation macro. Users need to adapt the corresponding build commands accordingly.

- New macro: `CATLASS_ARCH`, used to specify the target architecture. You can query its value in [SIMD BuiltIn Keywords](https://www.hiascend.com/document/detail/en/canncommercial/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_10053.html) (the `__NPU_ARCH__` column).
  - `Atlas A2 Training Series Products / Atlas A2 Inference Series Products`: `2201`
  - `Atlas A3 Training Series Products / Atlas A3 Inference Series Products`: `2201`
  - `Ascend 950PR/Ascend 950DT`: `3510`

- Related scenario descriptions:
  - `bisheng` command-line scenario: `bisheng ... -DCATLASS_ARCH=2201 ...`
  - `cmake` scenario: `add_compile_definitions(CATLASS_ARCH=2201)`
  - `msopgen/aclnn` project scenario:
    - Old usage: `add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 ...)`
    - New usage: `npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCATLASS_ARCH=2201)` (in an msopgen project, the first parameter defaults to `ascendc_kernels` and can be adjusted as needed)
  - CATLASS source repository: `bash scripts/build.sh -DCATLASS_ARCH=2201 ...`
  - Code reference in the library: [examples/CMakeLists.txt](https://gitcode.com/cann/catlass/blob/master/examples/CMakeLists.txt)

## Latest News

- [2026/04] Community edition [v1.5.0](https://gitcode.com/cann/catlass/releases/v1.5.0) released: added **Ascend 950** series examples, such as [Basic Matmul](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/43_ascend950_basic_matmul/README.md), [Flash Attention Inference](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/49_ascend950_flash_attention_infer/README.md), and [Per-Group & Per-Block Quant Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/51_ascend950_quant_matmul_per_group_per_block_tla/README.md); enhanced **TLA** capabilities, including `origin_shape`, `TileView`, and more; added [103 Dynamic W8A8 Per-Token Quantization](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/103_dynamic_optimized_quant_matmul_per_token_basic/README.md) to the [Matmul Generalization Project](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/102_dynamic_optimized_matmul/README.md).

- [2026/03] The community mainline officially started to add support for the next-generation Ascend hardware Ascend 950PR/Ascend 950DT.

- [2026/02] Community edition [v1.4.0](https://gitcode.com/cann/catlass/releases/v1.4.0) released, adding examples such as [StreamK Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md), [W4A4 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md), and [Sparse Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md).

- [2025/12] Community edition [v1.3.0](https://gitcode.com/cann/catlass/releases/v1.3.0) released, supporting [`FixPipe` Inline Quantization](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373), adding multiple templates to the [Matmul Generalization Project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md), and adding examples such as [INT4 Dequantization](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md) and [2D Convolution](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md).

- [2025/10] Community edition [v1.2.0](https://gitcode.com/cann/catlass/releases/v1.2.0) released, adding examples such as [Matmul Operator Generalization](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md).

- [2025/09] The CATLASS template library was officially open sourced.

See [CHANGELOG](CHANGELOG.md) for detailed updates in current and historical versions.

---

## 📌 Introduction

CATLASS (**CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**), known in Chinese as the Ascend Operator Template Library, is a code repository focused on providing base templates for high-performance matrix multiplication operators.  

CATLASS templates matrix operator code through layered abstraction. Therefore, it enables white-box assembly of operator compute logic and makes operator code reusable, replaceable, and partially modifiable. It is designed for Ascend hardware characteristics and supports complex pipeline layouts for operators such as `Flash Attention`. In addition, it shares upper-layer code logic while supporting specialization for differences in underlying hardware.

The template library enables fast development for custom scenarios. It provides performance optimization modules for different scenarios, so developers can assemble and customize them. Under custom shapes, its performance can reach 0.98 to 1.2 times the benchmark performance of the corresponding operator.

<div align="center">

<img src="docs/en/figures/Matmul.png" alt="Matmul Performance Comparison" style="width: 70%; max-width: 700px;">

</div>

<div align="center">

<img src="docs/en/figures/Grouped.png" alt="GroupedMatmul Performance Comparison" style="width: 90%; max-width: 900px;">

</div>

This repository is the co-created repository for CATLASS. It combines the strengths of the Ascend ecosystem to jointly design and develop operator templates, and provides high-performance implementation code examples for typical operators. For an overview, see [here](./docs/en/2_Design/00_project_overview.md#catlass-project-introduction).

## ⚡️ Quick Start

To quickly try CATLASS operator development and usage, see the following content.

- [Quick Start](./docs/en/1_Practice/01_quick_start.md): Quickly get started with the template library, and compile and run existing operator examples.

- [Basic Development Guide](./docs/en/1_Practice/02_host_example_assembly.md): Uses the basic Matmul operator as an example to introduce CATLASS-based operator development practices.

- [Developer Practices](./docs/en/README.md#1-practice): Provides practice examples from writing code at each operator layer to compilation and testing, then to Tiling tuning and operator optimization, from beginner to advanced levels.

## 📚 Advanced References

The following materials can help you further develop and tune CATLASS operators and implement GEMM-class operators with better performance.

- [CATLASS API](./docs/en/README.md#3-api): Introduces the layered features of CATLASS and the general matrix multiplication GEMM API.

- [CATLASS Design Summary](./docs/en/README.md#2-design): Summarizes documents such as example algorithm design, swizzle strategies, and TLA design in the CATLASS project.

## 📁 Directory Structure Description

The key directories are as follows. For the detailed directory structure, see [Project Directory](docs/en/2_Design/00_project_overview.md#project-directory).

```bash
catlass
├── cmake                     # cmake project files
├── docs                      # Documentation directory
├── examples                  # Root directory for kernel operator examples
|   ├── 00_basic_matmul       # Single-operator example
|   |   ├── basic_matmul.cpp  # Host-side operator invocation
|   |   ├── CMakeLists.txt
|   |   └── README.md         # Operator description example
|   ├── ...  
|   └── python_extension      # Project component for calling CATLASS operators from Python
├── include                   # Template header file set
|   ├── catlass               # Operator implementation logic at different layers
|   └── tla                   # Basic data structures related to computation
├── scripts                   # Build scripts
|   └── build.sh              # Operator example build script
├── tests                     # Test cases
└── tools                     # Related tools
    └── tuner                 # Tiling auto-tuning tool

```

## 💻 Software and Hardware Requirements

CATLASS depends on the following software and hardware environments:

- Ascend products:
  - [Atlas A2 Training Series Products / Atlas A2 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
  - [Atlas A3 Training Series Products / Atlas A3 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
  - Ascend 950PR/Ascend 950DT
- CPU architecture: `aarch64`/`x86_64`
- System: Linux supported by CANN (perform a [compatibility query](https://www.hiascend.com/hardware/compatibility))
- Software dependencies:
  - `gcc` >= 7.5, < 13.0
  - `cmake` >= 3.16
  - `python` >= 3.8, < 3.12

The hardware platforms supported by different CATLASS releases and the required minimum [CANN](https://www.hiascend.com/developer/download/community/result?module=cann) versions are shown in the following table:

| CATLASS Community Version                                                                                             | Minimum Supported CANN Package Version                                                                                                                                                                                                    | Supported Ascend Products                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Current                                                                                                               | [8.5.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)<br>[9.0.0.beta2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) (Ascend 950PR/Ascend 950DT)     | [Atlas A2 Training Series Products / Atlas A2 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) <br>[Atlas A3 Training Series Products / Atlas A3 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)<br>Ascend 950PR/Ascend 950DT |
| [v1.5.0](https://gitcode.com/cann/catlass/releases/v1.5.0)                                                            | [8.2.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)<br>[9.0.0.beta2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) (Ascend 950PR/Ascend 950DT) | [Atlas A2 Training Series Products / Atlas A2 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) <br>[Atlas A3 Training Series Products / Atlas A3 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)<br>Ascend 950PR/Ascend 950DT |
| [v1.4.0](https://gitcode.com/cann/catlass/releases/v1.4.0)—[v1.2.2](https://gitcode.com/cann/catlass/releases/v1.2.2) | [8.2.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)                                                                                                                                          | [Atlas A2 Training Series Products / Atlas A2 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) <br>[Atlas A3 Training Series Products / Atlas A3 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)                              |
| [v1.2.1](https://gitcode.com/cann/catlass/releases/v1.2.1)—[v1.0.0](https://gitcode.com/cann/catlass/releases/v1.0.0) | [8.2.RC1.alpha002](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002)                                                                                                                        | [Atlas A2 Training Series Products / Atlas A2 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) <br>[Atlas A3 Training Series Products / Atlas A3 Inference Series Products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)                              |

The following environments have been tested and support building [current CATLASS](https://gitcode.com/cann/catlass):

| System                                  | `CANN`      | `gcc` | `cmake` | `python` |
| --------------------------------------- | ----------- | ----- | ------- | -------- |
| Ubuntu 20.04.5                          | 8.5.0       | 9.3   | 3.16    | 3.10     |
| Ubuntu 22.04.5                          | 8.5.0       | 11.3  | 3.22    | 3.10     |
| openEuler 22.03 SP4                     | 8.5.0       | 10.3  | 3.22    | 3.10     |
| Ubuntu 22.04.5 (Compiling 950 Examples) | 9.0.0.beta2 | 11.3  | 3.22    | 3.10     |

## 👥 Collaborators

### [South China University of Technology Professor Lu Lu's Team](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### iFLYTEK Research Institute Engineering Group

## 📝 Related Information

- [Contribution Guide](CONTRIBUTING.md)
- [Security Statement](SECURITYNOTE.md)
- [License](LICENSE)
