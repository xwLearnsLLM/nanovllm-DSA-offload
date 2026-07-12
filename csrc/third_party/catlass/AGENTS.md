# AGENTS.md — CATLASS

> **CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**

昇腾算子模板库，提供高性能矩阵乘类算子基础模板与示例。

## Available Skills

| Skill                           | Description                                               | Location                                        |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| `catlass-example-to-pytest`     | 将 numbered CATLASS examples 接入 tests/optest 测试框架。 | `.agents/skills/catlass-example-to-pytest/`     |
| `catlass-example-to-torch-intf` | 将 catlass 示例迁移为 PyTorch extension 接口。            | `.agents/skills/catlass-example-to-torch-intf/` |

## 目录快速导航

| Path               | Description                            |
| ------------------ | -------------------------------------- |
| `examples/`        | 算子示例源码（00～120+）               |
| `include/catlass/` | 模板头文件（Kernel / Block / Tile）    |
| `docs/zh/`         | 中文文档：实践指南、设计总结、API 参考 |

详细文档见 [README.md](README.md)。
