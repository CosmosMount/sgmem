# SGMem：多人对话记忆检索研究

项目已实现 GroupMemBench 的内容检索基线（BM25、E5 Dense）、本地模型问答与判分、逐题结果及人工失败分析。主基准和扩展基准分别以子模块固定；SGAR 方法、正式实验结果和报告尚未完成。上游使用 GPT-5 与 `text-embedding-3-large`，本项目使用本地模型和 E5，因而是**官方流程的本地适配**，不是官方模型配置的精确复现。

2026-11-05：完成 Qwen3-8B + E5-small-v2 的双领域小样本试跑及逐条错误分析，结果、复跑方式和验证边界见 [Part 1/2 交接](docs/part12-pilot.md)。

## 安装与运行

需要 Python 3.10 或更高版本、Git 子模块，以及一个本地 OpenAI 兼容的 chat-completions 服务；服务需支持答题和 judge。Dense 还需要 E5 模型文件及 Sentence Transformers。容器运行方式见 [HPC 部署](docs/hpc-deployment.md)。

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

每次运行输出 `manifest.json`、`predictions.jsonl`、`metrics.json`。逐题记录包含参考答案，只供判分和后续分析；检索器仅接收提问者 ID 和问题文本。指标包括总体、六类、领域及领域内类别的准确率和分母。客户端提供终止原因时，还记录回答/判分的 length 截断次数。judge 输出无法解析时，官方汇总口径将其计入分母且不计正确，同时记录 `unclear`；服务调用失败单独记录，受影响的准确率设为不可用，不静默当成错误回答。当前没有实现自动重试/恢复、完整 GPU-hours 统计或上下文 token 截断控制；模型上下文限制、批量执行和资源测量需在正式运行前确认。

开发集和评估集按问题 ID、领域与类别划分，默认 25%/75%；它们仍共享历史对话，不能据此声称跨对话泛化。先在 dev 进行失败分析，方法冻结后再运行 `--partition test`。如果服务中断，已完成的逐题结果仍保存在输出目录；判分调用失败时也会保留已生成的回答，但当前需要另建目录重跑。切勿将不完整运行当作完整结果。

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

评分细则、50 GPU-hours 等信息来自用户提供的材料；实施前仍应对照原始课程说明核验。本机已使用 Qwen2.5-0.5B、E5 和 Finance 域的单题完成两条基线的连通性验证；这是小样本测试，不是正式实验结果，不能据此推断准确率或方法提升。
