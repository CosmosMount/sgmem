# Part 1 / Part 2 小样本验证与交接

日期：2026-11-05。任务范围是快速跑通、必要完善和错误分析。本次完成的是可检查、可继续扩展的小样本 Part 1/2，不是全部课程评估，也没有实现 Part 3。

## 已验证的结果

在金融服务器使用空闲 GPU 7 运行已有 Qwen3-8B，同时作为回答和判分模型；关闭 thinking。Dense 使用 E5-small-v2。Finance、Technology 各使用完整 30,000 条消息建立索引；从固定 dev 划分中每个领域、每个题型取 1 题，共 12 个问题，两种方法使用完全相同的问题和 top-k=10，共 24 条结果。

| 题型 | BM25 正确/总数 | Dense 正确/总数 |
|---|---:|---:|
| multi_hop | 0/2 | 1/2 |
| knowledge_update | 0/2 | 0/2 |
| temporal | 0/2 | 0/2 |
| user_implicit | 1/2 | 2/2 |
| term_ambiguity | 0/2 | 1/2 |
| abstention | 1/2 | 1/2 |
| **总计** | **2/12，16.67%** | **5/12，41.67%** |

Finance 分别为 BM25 2/6、Dense 1/6；Technology 分别为 0/6、4/6。差异来自很小的开发样本，不具有稳健性或显著性结论，不能当成完整 benchmark 分数。

24 条记录全部完成，48 次回答/判分调用均正常以 EOS 结束，没有生成截断、无法解析判分或调用失败。主试跑共 285.65 秒，约 0.0794 单卡 GPU-hours；PyTorch 峰值 allocated 显存约 15.97 GiB（不是整卡总显存占用）。试跑结束已释放 GPU。最终代码另用 2 条记录做过真实回归检查，用时 41.7 秒，两条均正确，逐条回答和提示词摘要与主试跑对应样本一致。

这证明了本地适配流程能真实运行。较低分数主要来自已定位的召回和回答问题，以及小模型 judge 的不稳定；没有发现本次运行的服务失败或答案截断。它不证明所有代码无缺陷，更不是官方 GPT-5 配置的精确复现。

## Part 2：错误分析

全部 17 条自动判错记录均已逐条检查原始消息与检索结果，标注作者明确记为 **Codex AI-assisted source review; pending teammate review**，不是人工专家金标。2 条 F9 是判分争议，保留原始分数，待队友复核。其余 15 条暂归入任务错误：

| 主原因 | 数量 | 解释 |
|---|---:|---|
| F5 时间/规则更新 | 4 | 当前规则或范围变化的消息缺失；其中一条还存在 QA 未充分使用已召回更新的情况 |
| F7 其他召回错误 | 8 | 明确证据不在 top-10，或相关片段不足以支持回答 |
| F8 QA 推理错误 | 3 | 日期/相对日期证据已提供但未理解，或没有明确指派时推断了负责人 |
| F9 判分争议（单列） | 2 | 核心回答与来源一致，但附加说明导致 judge 判错；原始 accuracy 不变 |

次级标签中，F2 群组干扰出现 1 次，F4 回复上下文缺失出现 2 次；会与主标签重叠，不能相加当成独立失败数。F1 说话人、F3 观点归属、F6 跨参与者推理保留在分类体系中，本轮没有足够证据作相应归因。题型名称不是错误原因，未强行填满六类失败。

可用于报告的代表案例：

- **更新规则未召回：** Finance/knowledge_update_1 的 Msg_7489、Msg_7491 要求扩大 UX 审批范围，覆盖短信失败后切换设备和恢复状态；两种检索都未召回。Technology/knowledge_update_2 的 Msg_4094 改为 explicit conflict resolution；BM25 只找到了回复它的 Msg_4165，回答仍遗漏更新，属于混合错误。
- **回复链可提供具体改进线索：** Technology/multi_hop_2 的正确规则见 Msg_847，存在 847 → 832 → 638 的回复链；BM25 已召回 Msg_638。有限的相邻回复扩展值得在后续 dev 实验检验。
- **已有证据没有被理解：** Finance/temporal_1 中 BM25 已召回带 2025-07-14 时间戳、正文要求 today 完成锁定的 Msg_11628，模型仍称日期不明确。Technology/temporal_2 中 Msg_6810 的时间是 2025-07-23，正文 Friday 对应 2025-07-25，模型未完成换算。这两例仅重排不一定有效。
- **群组干扰但不能用答案反推群组：** Technology/term_ambiguity_5 的 BM25 top-10 中，8 条来自 EmergencyResponseAgent、2 条来自 CodeReviewAgent，同名 Deployment Planning 混杂。问题未给出正确群名，相关消息的 target 与最终确认 deadline 也存在表述差异，故只作为混合失败/次级 F2，不能断言群组是唯一原因。
- **判分争议：** Technology/user_implicit_1 的 top-1 Msg_1511 与预测都给出 core signals only 或 user-facing alerts now 两项，预测还带了源消息中的重复通知风险，却被 judge 判错。另一个 abstention 样本明确说负责人未被命名，但因附加计划说明被判错。两条都只标 F9，不计入暂定的任务错误分布。

metadata_could_help 是分析者对后续方向的判断，不是实际修复成功率：8 条可能受益、4 条未知、5 条判断无直接帮助（含 2 条判分争议）。后续应先在固定 dev 上检验回复扩展、更新定位和群组区分，再冻结方法评估 test；不能把参考答案、人工证据 ID 或正确群组输入检索器。

## 代码完善

- scripts/run_local_baselines.py：可直接使用本地 Qwen 和 E5 跑成对基线，无需先启动 HTTP 服务；保留调用耗时、token 数、终止原因、配置和环境。
- src/sgmem/runner.py：增加显式 client_factory 接口，默认 HTTP 路径保持原行为。
- src/sgmem/qa.py：记录回答/判分的 finish_reason，统计 length 截断。使用同一个客户端时也保存回答当时的状态。
- src/sgmem/analysis.py：增加 F9 判分争议，与任务错误数量分开；禁止把 F9 与任务错误混合标注，不改变原始成绩。
- tests/test_baselines.py：覆盖原有流程、显式客户端入口、终止原因和 F9 边界，共 11 项通过。

原始 24 条结果产生于原提交 998f2a41d2776ae7bc5e6dbd2a594626b6c3a341，加一个临时直接加载模型的客户端适配；其实际驱动脚本作为 execution-driver.txt 保存在结果目录。随后把该行为整理为上述正式入口，并通过对应的真实样本回归。原始记录和分数没有为了新字段而重写。

试跑入口已在全量运行审核中统一更名为 `scripts/run_local_baselines.py`，原 24 条实验的配置与记录保持不变。

## 复跑方式与环境边界

服务器目录：/home/ai4s_group/XZG/NLP/sgmem。
使用现有 .venv-cpu；名称沿用最初 CPU 准备阶段，实际可使用 GPU。它复用已有环境的系统包，**不是独立 Docker 环境**。环境为 Python 3.10.20、torch 2.8.0、transformers 4.57.1、sentence-transformers 5.7.0、numpy 2.2.6、scikit-learn 1.7.2、rank-bm25 0.2.2。迁移服务器时重新准备环境；模型路径和 GPU 编号按新服务器修改。

```bash
cd /home/ai4s_group/XZG/NLP/sgmem
QWEN_MODEL=/home/ai4s_group/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
CUDA_VISIBLE_DEVICES=7 .venv-cpu/bin/python scripts/run_local_baselines.py \
  --model "$QWEN_MODEL" --embedding-model models/e5-small-v2 \
  --output results/runs/my-new-pilot
```

输出目录必须不存在。默认是 2 领域 × 6 类 × 1 dev 题，语料不下采样；更小的连通性检查可加 --domains Finance --categories user_implicit。CPU 可加 --device cpu，但 8B 模型速度会很慢。不加 --full 时仍做固定 dev 划分的小规模试跑；新版入口增加了 --full 和 --resume，详见 [全量运行方案](part1-full-benchmark.md)。

本机账号没有 Docker 执行权限，故本次没有验证同学的 Docker 镜像或真实 HTTP serving 集成；Dockerfile 和同学原有容器流程未改变。使用直接 Transformers 后端是为了复用已有环境、快速完成本次测试。

来源固定：
- 项目基线：998f2a41d2776ae7bc5e6dbd2a594626b6c3a341。
- GroupMemBench：461f00c3e8a1a4a3a2ed5814b729014502c4702f。
- Qwen3-8B：b968826d9c46dd6066d109eabc6255188de91218，复用服务器已有权重。
- E5-small-v2：ffb93f3bd4047442299a41ebb6fa998a38507c52，本次仅补充下载这一小型嵌入模型。

结果目录为 results/runs/quick-part12-20261105，含每种方法的 predictions.jsonl、metrics.json、manifest.json，以及 split.json、pilot-config.json、generation-audit.jsonl、failure-annotations.jsonl、failure-analysis.json、review-evidence.jsonl。原始输出可重新汇总；证据摘要只供分析使用。

```bash
.venv-cpu/bin/python -m pytest -q
.venv-cpu/bin/python -m sgmem analyze \
  --records results/runs/quick-part12-20261105/bm25/predictions.jsonl results/runs/quick-part12-20261105/dense/predictions.jsonl \
  --annotations results/runs/quick-part12-20261105/failure-annotations.jsonl \
  --output /tmp/sgmem-analysis-review.json
```

正式课程交付仍需扩大评估规模、人工复核错误标注与 judge、完成方法实验及报告/海报。本次证据可支持下一阶段实施判断，不能据此声称方法显著提升或完整作业已达成。
