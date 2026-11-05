# Part 1 全量运行审核与固定方案

日期：2026-11-05。用户已确认使用金融服务器 GPU 7，运行 GroupMemBench 四个领域全部问题。本文记录启动前审核和固定方案；实际完成状态以运行目录的 `run-config.json`、`benchmark-summary.json` 为准。

## 审核结论

课程 Part 1 要求 BM25 与 Dense、官方评测流程、总体和六类准确率。课程允许本地 Qwen 4B–8B 和 E5；因此本项目是官方流程的本地模型适配，不是 GPT-5/text-embedding-3-large 原配置的精确复现。

- 固定 GroupMemBench 提交 `461f00c3e8a1a4a3a2ed5814b729014502c4702f`。
- 四域共 120,000 条消息，全部 ID 在各自领域内唯一，reply-to 无悬空目标。逐条核对后，120,000 条消息的顺序和 QA 展示格式与上游相同。
- 全部 715 个问题逐题对比上游 BM25 实现，top-10 消息 ID、顺序和分数全部一致。
- QA 与 judge 的提示词、512/256 输出预算、Final 提取和官方严格 Correct 判定，通过直接调用上游函数的测试核验。gold 只进入 judge，不进入 QA 或检索。
- E5 使用 passage/query 前缀、单位归一化、精确点积余弦检索；稳定排序明确处理同分项。E5 与上游商用嵌入模型不同，不能声称两者向量或检索结果一致。
- 官方问题只有 id/question/answer/asking_user_id，没有提问时点。全域历史可见性沿用上游，不自行添加按题目日期截断或使用答案反推群组。
- 原先的短测入口缺少全量模式和安全恢复，现已补齐。26 项自动测试通过，含上游流程对照、配置与输入变更拒绝、失败重试、截断写入恢复和 dev/test 汇总。
- 真实 Qwen/E5 回归覆盖 Finance、Healthcare、Manufacturing，共 6 条记录；全部正常完成，无调用失败。评分为正常模型输出，不以正确率作为代码连通性的通过条件。

审核支持启动完整本地基线。有限测试不能证明程序在所有情形下无缺陷；尤其本地 Qwen 同时充当 judge 的判分噪声仍需报告和人工抽查。

## 范围与固定配置

| 领域 | 消息数 | 问题数 | 两方法结果数 |
|---|---:|---:|---:|
| Finance | 30,000 | 214 | 428 |
| Technology | 30,000 | 213 | 426 |
| Healthcare | 30,000 | 130 | 260 |
| Manufacturing | 30,000 | 158 | 316 |
| 合计 | 120,000 | 715 | 1,430 |

使用种子 42、按领域和题型分层的原 25%/75% question-level 划分：dev 179 题，test 536 题。Finance/Technology 的划分与第一次试跑一致。两组仍共享对话语料，不代表跨对话泛化。先固定基线配置再运行；后续方法只能用 dev 选择，不能根据 test 失败修改方法。全体 715 题的结果与 test-only 结果分别汇总。

- QA/judge：同一份已有 Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- Dense：同一份 E5-small-v2，revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`。
- 内容检索、每题 top-k=10，元数据仅在展示阶段附加；没有引入 Part 3 重排。
- Qwen 非 thinking、BF16、SDPA、temperature=0.2、top_p=1、top_k=0，每次调用固定种子 42。
- 每个领域分别建立索引，两个方法各运行 715 题；共 2,860 次正常的 QA/judge 调用，不含测试和失败重试。
- 不下载新的 QA 模型，不调用付费 API；本机 Python/Transformers 后端，不宣称 Docker 验证完成。

## 启动与恢复

统一入口从 `run_local_pilot.py` 更名为 `run_local_baselines.py`；默认仍是少量 dev 题。`--full` 运行所选领域和类别的全部 dev/test 题，并强制代码和上游工作树干净。输出目录不允许覆盖。

```bash
cd /home/ai4s_group/XZG/NLP/sgmem
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-cpu/bin/python -u scripts/run_local_baselines.py \
  --model /home/ai4s_group/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --model-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --embedding-model models/e5-small-v2 \
  --embedding-revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --domains Finance Technology Healthcare Manufacturing --full \
  --output results/runs/part1-full-20261105
```

中断后使用完全相同命令追加 `--resume`。新 schema 的运行支持恢复；早期 pilot 的结果不直接续跑。成功题目跳过，失败题目保留旧 attempt 并重试。损坏的最后一个未写完记录先保留原始字节，再从最后成功题恢复；中间记录损坏则报错，禁止静默忽略。代码、输入内容、划分或配置变化会拒绝恢复。恢复前确认同一任务没有仍在运行；不要同时运行两个写入相同输出目录的进程。

每题写入 `attempts.jsonl` 并同步到磁盘。`progress.json` 提供预期数量、已完成数量和最后题目；`predictions.jsonl` 是正常退出或异常收尾时生成的每题最终记录。`metrics.json` 的 `complete` 和 `expected_questions` 用于区分部分结果与完整结果。runner 遇到模型调用错误立即停止，不继续生成大量失败记录。

`run-config.json` 保存软件、模型配置、权重文件大小、revision、代码版本、硬件、实际耗时和 GPU-hours；`generation-audit.jsonl` 保存每次调用的题目、阶段、提示词摘要、token 数和终止原因。GPU-hours 按本任务单卡占用 wall time 计算，包含模型加载和索引期间的驻留，不等于利用率积分。进程被强制 SIGKILL 时最后一个 session 资源收尾可能缺失，需要结合启动日志补记；不能当作零消耗。

最终 `benchmark-summary.json` 保留两个方法的总体、六类、领域、领域内类别和 dev/test 统计，并验证两种方法覆盖相同题目。保留调用错误、无法解析判分和截断计数，不用人工复核结果静默替换原始分数。
