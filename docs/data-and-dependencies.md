# 数据、上游代码与产物管理

## 上游与数据

[GroupMemBench](https://github.com/UCSB-NLP-Chang/GroupMemBench) 为首要基准，已作为 `external/GroupMemBench` 子模块导入；父仓库的 gitlink 固定所用提交。上游 README 声明代码采用 MIT 许可，对话数据和问题集采用 CC BY 4.0，使用和分发时须遵循各自条款。[EverMemBench](https://github.com/EverMind-AI/EverMemBench) 已作为 `external/EverMemBench` 子模块导入，用于计划中的扩展基准；数据需按上游说明单独获取。当前固定提交的仓库根目录和 README 未发现代码许可证，复用或分发代码前须向上游核实；数据许可需单独核验。其上游评测依赖与本项目的本地模型方案不同，运行前须确认适配方式。适配代码放在 `src/sgmem/`，不直接改动上游检出。

克隆项目后执行 `git submodule update --init --recursive`；或使用 `git clone --recurse-submodules <仓库地址>`。更新上游时，显式检出并审查目标提交，在父仓库提交 gitlink 的变更，并核验数据字段、任务规则、评测行为与许可；不要自动跟随上游最新版本。仅在完整性验证确有需要时记录校验值。

原始数据 `data/raw/` 只读；转换结果在 `data/processed/`，索引、模型和运行缓存均可重建。split 清单只有在许可允许且不含敏感数据时才纳入版本控制，否则存于共享位置并记录获取方式。测试样例使用人工构造的最小数据，避免未经许可复制真实样本。

## 参考项目与运行依赖

[SpeakerMem-R1](https://github.com/2022hpsk/SpeakerMemR1)、[Graphiti](https://github.com/getzep/graphiti)、[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) 和 [Mem0](https://github.com/mem0ai/mem0) 用于研究方法和相关工作参考；当前不作为运行依赖或子模块。确需直接复用或比较时，再核查许可并按实际需要引入。`rank-bm25`、`sentence-transformers`、FAISS、Transformers 和 vLLM 是候选软件依赖，确定使用后通过包管理器固定，不把它们的源码仓库作为子模块。

## 依赖与配置

Python、包及推理后端版本应在首个可执行骨架完成后固定于 `pyproject.toml` 和锁文件；记载安装与运行说明。配置和凭据分开：仓库只放无秘密的示例，不提交 `.env`、访问令牌或个人路径。各成员采用相同模型 revision、tokenizer、评测器版本和解析后的配置；不同硬件或模型更改需明确记录。

## 存储与恢复

`results/runs/`、数据、模型权重、日志和缓存默认不纳入 Git。团队须选定可访问的共享存储、权限和备份规则，并为正式结果记录 URI、必要的运行信息与保留期限。未经许可不上传受限数据、模型或敏感轨迹。可入库的报告图表应能追溯到 run ID 和生成脚本，避免把手工修改的图当作实验产物。
