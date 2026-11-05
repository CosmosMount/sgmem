# SGMem：多人对话记忆检索研究

项目已完成 GroupMemBench 的 BM25/E5 Dense 全量本地基线和 Part 2 结构性失败分析。四域共 715 题，两种方法共 1,430 条结果；BM25 32.73%，Dense 28.53%。Part 2 覆盖固定开发样本 24 道题、38 条判错记录和 61 条源消息，标注来源明确为 AI-assisted。SGAR 修复方法和最终课程报告/海报仍需完成。上游使用 GPT-5 与 `text-embedding-3-large`，本项目使用本地 Qwen3-8B 和 E5，因而是**官方流程的本地适配**，不是官方模型配置的精确复现。

**从这里接手：[Part 1/2 全量结果与错误分析](docs/part12-full-results.md)。** [仓库内精简证据](docs/evidence/part12-full-20261105/)可在无 GPU 环境复核：

```bash
python scripts/verify_part12_evidence.py
# 可选：用固定子模块同时核验源消息和回复图
python scripts/verify_part12_evidence.py --benchmark external/GroupMemBench
```

2026-11-05：完成 Qwen3-8B + E5-small-v2 的双领域小样本试跑及逐条错误分析，结果、复跑方式和验证边界见 [Part 1/2 交接](docs/part12-pilot.md)。

全量运行入口、715 题审核和恢复方法见 [Part 1 全量运行方案](docs/part1-full-benchmark.md)。

## 安装与运行

需要 Python 3.10 或更高版本与 Git 子模块。已有本地 Qwen 权重时，可按[全量运行方案](docs/part1-full-benchmark.md)直接使用 Transformers，无需 HTTP 服务；也可使用下述 OpenAI 兼容 chat-completions 服务。Dense 需要 E5 模型文件及 Sentence Transformers。容器运行方式见 [HPC 部署](docs/hpc-deployment.md)，本次金融服务器的全量结果来自直接 Transformers 后端。

```bash
git submodule update --init --recursive
# 本机已准备 .venv；新机器应先创建自己的 Python 虚拟环境
source .venv/bin/activate
python -m pip install -e '.[dense,serving,test]'
python -m pytest
```

### 本机 Qwen 服务

本机已经准备 `.venv` 和 `models/e5-base-v2`。本次 `.venv` 使用 Python 3.10 的 `--system-site-packages`，复用宿主机已有的 GPU 版 PyTorch，因此不是完全独立的环境；Docker 使用独立的 Python 3.11 和固定依赖；`sgmem:local` 已通过本机 RTX 3090 可见性和容器内单题流水线检查。当前机器缓存了 Qwen2.5-0.5B-Instruct，可用于连通性验证（它不适合作为正式结果的质量保证）。

在项目根目录打开第一个终端，按本机现有缓存启动服务：

```bash
source .venv/bin/activate
export QWEN_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
HF_HUB_OFFLINE=1 NO_PROXY=127.0.0.1,localhost transformers serve "$QWEN_MODEL" \
  --host 127.0.0.1 --port 8000 --device cuda --dtype float16
```

在第二个终端同样激活 `.venv` 并设置 `QWEN_MODEL` 与 `NO_PROXY=127.0.0.1,localhost`；客户端模型名必须与服务加载的模型一致。正式实验可替换为选择并固定的更合适模型；若使用不同机器，请修改模型路径和设备。以下命令先生成固定的按题目分层的划分：

```bash
python -m sgmem split --output data/splits/finance-tech.json
```

在开发集运行基线，两个命令使用相同模型、问题、`top-k` 和评测规则。输出目录必须不存在，避免覆盖既有实验：

```bash
python -m sgmem run --method bm25 --answer-model "$QWEN_MODEL" \
  --split data/splits/finance-tech.json --partition dev --output results/bm25-dev
python -m sgmem run --method dense --answer-model "$QWEN_MODEL" --device cuda \
  --embedding-model models/e5-base-v2 \
  --split data/splits/finance-tech.json --partition dev --output results/dense-dev
```

本机 E5 文件来自 `intfloat/e5-base-v2` 的 `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` 修订；正式实验应固定模型和数据来源并记录环境。使用模型仓库名称而非本地路径时，可通过 `--embedding-revision` 指定修订。若系统设置了代理，请对客户端设置 `NO_PROXY=127.0.0.1,localhost`。默认 API 地址为 `http://127.0.0.1:8000/v1`，可通过 `--base-url` 更改，`--judge-model` 默认与答题模型相同；如服务需要密钥，从环境变量 `SGMEM_API_KEY` 读取。用 `--categories multi_hop --domains Finance --limit 2` 可进行小样本调试，完成后删除临时输出。实际考试或课程要求如指定其他切分，以其要求为准。

每次运行输出 `manifest.json`、`predictions.jsonl`、`metrics.json`。逐题记录包含参考答案，只供判分和后续分析；检索器仅接收提问者 ID 和问题文本。指标包括总体、六类、领域及领域内类别的准确率和分母。客户端提供终止原因时，还记录回答/判分的 length 截断次数。judge 输出无法解析时，官方汇总口径将其计入分母且不计正确，同时记录 `unclear`；服务调用失败单独记录，受影响的准确率设为不可用，不静默当成错误回答。本地入口支持全量模式、显式断点恢复和单卡 GPU-hours 记录；不会自动循环重试。它检查模型上下文上限并报错，不静默截断 QA 输入。HTTP 服务的模型上下文上限仍由服务端控制。

开发集和评估集按问题 ID、领域与类别划分，默认 25%/75%；它们仍共享历史对话，不能据此声称跨对话泛化。先在 dev 进行失败分析，方法冻结后再运行 `--partition test`。如果服务中断，逐题 attempt 和已生成回答会保留；修复原因后，用相同命令加 --resume 续跑，配置/输入/代码变化会被拒绝。切勿将不完整运行当作完整结果。

## 人工失败分析

从开发集运行生成未标注模板，再人工填写 `annotator`、`failure_types`、`primary_failure_type`、`error_stage`、证据 ID 和分析说明。标签 `F1`–`F8` 分别覆盖说话人、群组、观点归属、回复上下文、时间更新、跨参与者推理、其他检索和 QA 推理错误。另加 `F9` 表示待复核的判分争议，必须单独标注并设置 `error_stage=evaluation`；它不修改原始准确率，在汇总中与任务错误分开。多标签可以共存（F9 除外），主标签只能选一个；分析流程不会自动猜测错误原因或支持证据。

```bash
python -m sgmem annotate --records results/bm25-dev/predictions.jsonl results/dense-dev/predictions.jsonl \
  --output data/annotations/dev-failures.jsonl
# 人工编辑 data/annotations/dev-failures.jsonl
python -m sgmem analyze --records results/bm25-dev/predictions.jsonl results/dense-dev/predictions.jsonl \
  --annotations data/annotations/dev-failures.jsonl --output results/dev-analysis.json
```

汇总同时给出已判明失败、未判明、已标注和未标注数量、主标签及多标签分布、类别/方法分布、人工证据覆盖和有说明的案例。未解析的 judge 不自动进入失败案例，因此失败分析分母与官方准确率分母可能不同；检查标注抽样方式与分母后再用于报告。题目、答案和对话数据的共享需遵循上游许可。

## 项目文档

- [Agent 工作约定](AGENTS.md)和[团队协作协议](CONTRIBUTING.md)
- [架构与接口设计](docs/architecture.md)、[实验协议](docs/experiment-protocol.md)
- [HPC 容器部署与 Qwen 升级](docs/hpc-deployment.md)、[检索参考](docs/retrieval-references.md)
- [实施路线图](docs/roadmap.md)、[后续 CI 清单](docs/ci-plan.md)、[数据和依赖](docs/data-and-dependencies.md)

评分细则与 50 GPU-hours 上限来自用户提供的课程材料。本次全量运行累计 3.5907 GPU-hours；早期 Qwen2.5-0.5B 连通性测试和 Qwen3-8B 试跑属于历史准备，不应替代全量结果。当前报告没有声称完成 Part 3 修复或取得修复后的准确率提升。
