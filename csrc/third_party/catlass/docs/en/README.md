# CATLASS Project Documentation

## 1. Practices

> Code practices that guide developers through the steps of using and developing at each level of CATLASS, gradually building the ability to perform complete operator development, testing, tuning, and model integration.

- [01_quick_start](./1_Practice/01_quick_start.md): Introduces environment setup for the template library and how to build and run the provided operator samples.
- [02_host_example_assembly](./1_Practice/02_host_example_assembly.md): Explains host-side Matmul assembly.
- [03_kernel_development](./1_Practice/03_kernel_development.md): Provides a kernel code breakdown, demonstrating mechanisms such as template assembly, arguments, params, and key functions.
- [04_block_mmad_development](./1_Practice/04_block_mmad_development.md): Provides a block_mmad code breakdown, including template assembly mechanisms and major interface descriptions.
- [05_block_scheduler_development](./1_Practice/05_block_scheduler_development.md): Provides a block_schedule code breakdown, including template assembly mechanisms and major interface descriptions.
- [06_tile_development](./1_Practice/06_tile_development.md): Provides a tile_copy/tile_mmad code breakdown, including template assembly mechanisms and major interface descriptions.
- [07_epilogue_adaptation](./1_Practice/07_epilogue_adaptation.md): Covers host/kernel layer epilogue adaptation for GEMM operators, as well as block/tile development for epilogues.
- [08_evaluation](./1_Practice/08_evaluation.md): Covers the use of debugging and profiling tools for precision issue locating and performance bottleneck analysis.
- [09_example_contribution_guide](./1_Practice/09_example_contribution_guide.md): Details the complete process for sample design, development, testing, and integration.
- [10_innovative_example_development_guide](./1_Practice/10_innovative_example_development_guide.md): Provides guidance on the development process of innovative samples.
- [11_matmul_optimization](./1_Practice/11_matmul_optimization.md): Introduces basic tuning methods in the template library, including how to achieve performance gains through tiling parameter adjustment and applying different dispatch policies.
- 12_example_integration: Introduces sample adaptation and integration into the whole network (to be contributed).
- evaluation (folder): Debugging-related
  - [ascendc_dump](./1_Practice/evaluation/ascendc_dump.md)
  - [msdebug](./1_Practice/evaluation/msdebug.md)
  - [performance_tools](./1_Practice/evaluation/performance_tools.md)
  - [print](./1_Practice/evaluation/print.md)
  - [precision_analysis_basics](./1_Practice/evaluation/precision_analysis_basics.md): precision analysis basics
  - [precision_debug](./1_Practice/evaluation/precision_debug.md): sample precision issue locating
  - [bottleneck_analysis_and_optimization](./1_Practice/evaluation/bottleneck_analysis_and_optimization.md): performance bottleneck analysis and optimization
- others (folder): internally and externally contributed practice documents that are difficult to categorize
  - tla_rebuild: TLA sample refactoring (to be contributed)
  - [migration_from_atlasA2_to_Ascend950_guideline](./1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md): Recommended solution for the compatibility of existing operators on the Atlas A2 platform with Ascend 950
  - [conv_kernel_development](./1_Practice/others/conv_kernel_development.md): Conv operator development guide
  - [conv_kernel_optimization](./1_Practice/others/conv_kernel_optimization.md): Conv operator performance optimization
  - [FA_kernel_optimization](./1_Practice/others/FA_kernel_optimization.md): FA operator performance optimization
  - [fused_kernel_optimization](./1_Practice/others/fused_kernel_optimization.md): CV fused operator performance optimization cases
  - [kernel_execution](./1_Practice/others/kernel_execution.md): `<<<>>>` Direct calls on new operators

## 2. Design

- [00_project_overview](./2_Design/00_project_overview.md): Project introduction, layered modular design, and code repository structure design
- 01_kernel_design: Algorithm design
  - 00_basics (folder): CATLASS development basics
    - [atlasA2_hardware_info](./2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md): Atlas A2 hardware information
    - [atlasA2_gemm_instruction_set](./2_Design/01_kernel_design/00_basics/atlasA2_gemm_instruction_set.md): Hardware instruction set related to Atlas A2 GEMM samples
  - [01_example_design](./2_Design/01_kernel_design/01_example_design.md): Overview of sample design documents in the repository (each sample's document is placed in its own sample folder; this document only provides a summary and index).
  - [02_swizzle](./2_Design/01_kernel_design/02_swizzle.md): Basic introduction to the `Swizzle` policies in the template library, which affects the order of basic blocks on AI Cores.
  - [03_dispatch_policies](./2_Design/01_kernel_design/03_dispatch_policies.md): Introduction to `DispatchPolicy`, an important template parameter in `BlockMmad` at the `Block` layer.
  - [04_matmul_summary](./2_Design/01_kernel_design/04_matmul_summary.md): Introduction to the existing `matmul` template design in the `examples` directory of the template library, including the sample template list, theoretical template list, engineering optimization list, and brief introduction to template application. This document can be used as a reference for matmul performance tuning.
  - [05_aswt](./2_Design/01_kernel_design/05_aswt.md): Description of the adaptive sliding window tiling policy.
  - 06_quant_summary: Low-precision topics (to be contributed)
- 02_tla:
  - [01_layout](./2_Design/02_tla/01_layout.md): Layout structure and related interfaces for TLA
  - [02_layout_tag](./2_Design/02_tla/02_layout_tag.md): Layout tags such as RowMajor, ColumnMajor, zN, and nZ and related interfaces, that is, the legacy layout structure
  - [03_tensor](./2_Design/02_tla/03_tensor.md): Tensor structure
- 03_evg:
  - [01_evg_design](./2_Design/03_evg/01_evg_design.md): EVG positioning, layering, execution model, and graph organization
  - [02_evg_extension](./2_Design/03_evg/02_evg_extension.md): EVG extension conventions, describing when to add ComputeFn, when to add nodes, and the constraints to follow during implementation
  - [03_evg_quick_start](./2_Design/03_evg/03_evg_quick_start.md): Description on the EVG integration process using `Matmul + Add` as an example.

## 3. APIs

- [README](./3_API/README.md): API list
- [gemm api](./3_API/gemm_api.md): Gemm APIs
- [evg_api](./3_API/evg_api.md): Description on the EVG integration mode, parameter order, and common nodes
- [Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html): Ascend C API list

## Appendix

> External articles and videos

- [Q&A](./Q&A.md)
- Technical articles
  - Fundamentals
    - [C++ Template Explained](https://www.runoob.com/w3cnote/c-templates-detail.html)
    - [Ascend C operator development documentation](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
  - Concept understanding
  - Troubleshooting
  - Performance optimization
  - Best practices
- Training videos
  - [Ascend community online courses](https://www.hiascend.com/edu/courses): Get to know Ascend through structured course videos. Recommended courses:
    - [Ascend C Operator Development (Beginner)](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
    - [Ascend C Operator Development (Intermediate)](https://www.hiascend.com/developer/courses/detail/1696414606799486977)
    - [Ascend C Operator Development (Advanced)](https://www.hiascend.com/developer/courses/detail/1696690858236694530)
  - Ascend Training Camp on CATLASS
    - [Basic Concepts of CATLASS in One Stop](https://www.bilibili.com/video/BV1f1BDBMES2/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47): `Ascend CANN` [Code Power] Lecture 1 in CATLASS Learning Series. This lecture provides a comprehensive introduction to CATLASS, including its general design, quick start guide for operators, development overview, and community contribution.
    - [Hands-on CATLASS Operator Development](https://www.bilibili.com/video/BV1DmBhBNEu8/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47): `Ascend CANN` [Code Power] Lecture 2 in CATLASS Learning Series. Using the basic Matmul operators as an example, this video provides a comprehensive introduction to NPU-based matrix multiplication theoretical modeling and code implementation (at the host, kernel, and block layers).
