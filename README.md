# SGMem：多人对话记忆检索研究

本仓库目前处于**设计阶段**，已导入官方 GroupMemBench 和 EverMemBench 子模块，尚未实现本项目的数据加载、检索或实验流水线。目标是以 GroupMemBench 为主要基准，在统一的问答和评测条件下比较 BM25、Dense Retrieval 与轻量的 speaker/group-aware reranking（SGAR），用逐题轨迹分析错误，并测量质量与计算成本。EverMemBench 是有余力时的扩展。

## 从这里开始

- [Agent 工作约定](AGENTS.md)：代码、临时产物清理和文档规则。
- [架构与接口设计](docs/architecture.md)：模块边界、目标目录和数据契约。
- [实验协议](docs/experiment-protocol.md)：数据划分、对照、资源统计和结果追溯。
- [团队协作协议](CONTRIBUTING.md)：模块负责制、分支、PR 和合并顺序。
- [实施路线图](docs/roadmap.md)：可并行任务、依赖与验收标准。
- [后续 CI 验收清单](docs/ci-plan.md)：各自动检查的启用条件和通过标准。
- [外部依赖与数据管理](docs/data-and-dependencies.md)：版本固定、数据和产物存储。

## 当前状态与范围

本仓库是空仓库的设计起点；目标目录、运行命令、模型和指标均为计划，不代表流水线已实现、模型已下载或已有结果。建议优先完成 GroupMemBench 的小样本端到端流程，再扩展到完整实验。参考方案提出 Python 3.11、E5-base-v2、Qwen3-4B-Instruct-2507 与本地推理；最终选择须按实际硬件、模型许可及课程要求确认。

输入的两份文本转述了课程项目和评分材料，包括 50 GPU-hours 限制、六页报告及评分权重；本仓库未收到原始课程说明和评分 PDF，这些数字和约束在实施前应由团队对照原文确认。基准的数据字段、评测方式和可用划分也需在固定上游版本后核验。任何实验结果均不得预先假设提升。

## 初始化上游仓库

克隆后执行 `git submodule update --init --recursive`，检出父仓库记录的两个基准提交。详见[外部依赖与数据管理](docs/data-and-dependencies.md)。
