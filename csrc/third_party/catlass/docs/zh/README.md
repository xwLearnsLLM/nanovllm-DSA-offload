# CATLASS 项目文档

## 1 Practice

> 代码实践，指导开发者按步骤上手CATLASS各层级代码开发和使用，逐渐具备完整算子开发、测试、调优、模型使用的能力。

- [01_quick_start](./1_Practice/01_quick_start.md)：介绍模板库的环境准备，和提供的算子样例的编译执行。
- [02_host_example_assembly](./1_Practice/02_host_example_assembly.md)：host侧组装matmul讲解
- [03_kernel_development](./1_Practice/03_kernel_development.md)：kernel代码拆解，展示模板组装机制、arguments、params及关键函数等代码
- [04_block_mmad_development](./1_Practice/04_block_mmad_development.md)：block_mmad代码拆解，模板组装机制、主要接口说明
- [05_block_scheduler_development](./1_Practice/05_block_scheduler_development.md)：block_scheduler代码拆解，模板组装机制、主要接口说明
- [06_tile_development](./1_Practice/06_tile_development.md)：tile_copy/tile_mmad代码拆解，模板组装机制、主要接口说明
- [07_epilogue_adaptation](./1_Practice/07_epilogue_adaptation.md)：gemm算子的host/kernel层适配epilogue，及epilogue的block/tile开发
- [08_evaluation](./1_Practice/08_evaluation.md)：调测工具使用，精度问题定位，性能瓶颈分析
- [09_example_contribution_guide](./1_Practice/09_example_contribution_guide.md)：完整样例的设计、开发、测试、合入流程
- [10_innovative_example_development_guide](./1_Practice/10_innovative_example_development_guide.md)：创新样例的开发流程指引
- [11_matmul_optimization](./1_Practice/11_matmul_optimization.md)：介绍模板库下的基础调优方式，包括如何通过Tiling调参、应用不同的Dispatch策略的方式，快速获得性能提升。
- 12_example_integration：样例适配、接入到整网的方式（待贡献）
- evaluation（folder）：调测相关
  - [ascendc_dump](./1_Practice/evaluation/ascendc_dump.md)工具
  - [msdebug](./1_Practice/evaluation/msdebug.md)工具
  - [performance_tools](./1_Practice/evaluation/performance_tools.md)工具
  - [print](./1_Practice/evaluation/print.md)工具
  - [precision_analysis_basics](./1_Practice/evaluation/precision_analysis_basics.md)：精度分析基础
  - [precision_debug](./1_Practice/evaluation/precision_debug.md)：样例精度问题定位
  - [bottleneck_analysis_and_optimization](./1_Practice/evaluation/bottleneck_analysis_and_optimization.md)：性能瓶颈分析及优化手段
- others（folder）：存放内部和外部贡献的难以归类的实践文档
  - tla_rebuild：TLA样例改造（待贡献）
  - [migration_from_atlasA2_to_Ascend950_guideline](./1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md)：介绍推荐的AtlasA2平台存量算子向Ascend950代际兼容方案
  - [conv_kernel_development](./1_Practice/others/conv_kernel_development.md)：Conv类算子开发指南
  - [conv_kernel_optimization](./1_Practice/others/conv_kernel_optimization.md)：Conv类算子性能优化
  - [FA_kernel_optimization](./1_Practice/others/FA_kernel_optimization.md)：FA类算子性能优化
  - [fused_kernel_optimization](./1_Practice/others/fused_kernel_optimization.md)：CV融合算子性能调优案例集
  - [kernel_execution](./1_Practice/others/kernel_execution.md)：`<<<>>>`直调新开发算子

## 2 Design

- [00_project_overview](./2_Design/00_project_overview.md)：项目介绍、分层模块化设计、代码仓结构设计
- 01_kernel_design：算法设计
  - 00_basics（folder）：CATLASS开发前置基础知识
    - [atlasA2_hardware_info](./2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md)：AtlasA2 硬件信息
    - [atlasA2_gemm_instruction_set](./2_Design/01_kernel_design/00_basics/atlasA2_gemm_instruction_set.md)：AtlasA2 GEMM类样例相关硬件指令集介绍
  - [01_example_design](./2_Design/01_kernel_design/01_example_design.md)：库上样例设计文档一览（将各样例文档放到样例文件夹内，此处只做归纳、牵引）
  - [02_swizzle](./2_Design/01_kernel_design/02_swizzle.md)：对模板库中`Swizzle`策略的基本介绍，这影响了AI Core上计算基本块间的顺序。
  - [03_dispatch_policies](./2_Design/01_kernel_design/03_dispatch_policies.md)：对模板库在`Block`层面上`BlockMmad`中的一个重要模板参数`DispatchPolicy`的介绍。
  - [04_matmul_summary](./2_Design/01_kernel_design/04_matmul_summary.md)：对模板库的`examples`目录内已有的`matmul`模板设计进行介绍，包含样例模板清单、理论模板清单、工程优化清单、模板应用浅述，可用于matmul性能调优时参考。
  - [05_aswt](./2_Design/01_kernel_design/05_aswt.md)：自适应滑窗tiling策略说明
  - 06_quant_summary：低精度专题（待贡献）
- 02_tla：
  - [01_layout](./2_Design/02_tla/01_layout.md)：TLA设计的layout结构和相关接口说明
  - [02_layout_tag](./2_Design/02_tla/02_layout_tag.md)：RowMajor、ColumnMajor、zN、nZ等layoutTag介绍和接口说明，即旧版layout结构
  - [03_tensor](./2_Design/02_tla/03_tensor.md)：tensor结构体
- 03_evg
  - [01_evg_design](./2_Design/03_evg/01_evg_design.md)：EVG 的定位、分层关系、执行模型与图组织方式
  - [02_evg_extension](./2_Design/03_evg/02_evg_extension.md)：EVG 的扩展规范，说明何时加 ComputeFn、何时加节点，以及实现时的约束
  - [03_evg_quick_start](./2_Design/03_evg/03_evg_quick_start.md)：以 `Matmul + Add` 为例说明 EVG 的基础接入流程

## 3 API

- [README](./3_API/README.md)：API清单入口
- [gemm api](./3_API/gemm_api.md)：Gemm API
- [evg api](./3_API/evg_api.md)：EVG 的接入方式、参数顺序和常用节点说明
- [Ascend C API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html)：昇腾社区Ascend C API列表

## Appendix

> 外部开源文章、视频

- 常见问题 [Q&A](./Q&A.md)
- 技术文章
  - 基础入门
    - [C++ template详解](https://www.runoob.com/w3cnote/c-templates-detail.html)
    - [Ascend C算子开发文档](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)
  - 概念理解
  - 问题定位
  - 性能优化
  - 优秀实践
- 培训视频
  - [昇腾社区在线课程](https://www.hiascend.com/edu/courses)：通过体系化的课程视频学习昇腾。相关课程推荐：
    - [Ascend C算子开发（入门）](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
    - [Ascend C算子开发（进阶）](https://www.hiascend.com/developer/courses/detail/1696414606799486977)
    - [Ascend C算子开发（高级）](https://www.hiascend.com/developer/courses/detail/1696690858236694530)
  - 昇腾训练营 CATLASS相关特辑
    - [一站式掌握CATLASS模板库基本概念](https://www.bilibili.com/video/BV1f1BDBMES2/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47)：`昇腾CANN`【码力全开特辑】CATLASS学习系列课程第一讲。完整介绍CATLASS整体情况、算子快速上手、发展全景、生态共建。
    - [CATLASS算子开发初体验](https://www.bilibili.com/video/BV1DmBhBNEu8/?spm_id_from=333.1387.collection.video_card.click&vd_source=ae7f2ef56954c6e4a7397c8386a66b47)：`昇腾CANN`【码力全开特辑】CATLASS学习系列课程第二讲。以基础Matmul算子为例，完整介绍基于NPU的矩阵乘理论建模、代码实现（host/kernel/block三个层面）。
