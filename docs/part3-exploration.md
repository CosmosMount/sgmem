# Part 3 完整探索

本轮从 GitHub `5578d30` 开始，先在 `f725168` 提交探索计划，再实现和核验八个配置。目的在于从多种可解释的方向中选一个课程最终方法。核心检索共用 `src/sgmem/structured.py`，沿用现有 BM25、E5、回复图和问答流程，未增加训练组件或依赖。

## 固定矩阵

| 配置 | 实际改变 | 直接对照 |
|---|---|---|
| `bm25-fields` | 正文索引增加 role/group/phase/topic；保留原查询前缀，返回原消息 | BM25 |
| `bm25-scope` | 全文召回与范围内去掉字段名称后的正文召回做等权 RRF | BM25 |
| `bm25-person` | 在范围支路加明确目标人物/角色条件；普通提问者没有特殊优先级 | scope |
| `bm25-time` | 问题明确询问当前状态时，在已匹配范围/人物内加时间支路 | person |
| `hybrid` | 原 BM25 与原 E5 的等权 RRF | BM25、Dense |
| `hybrid-meta` | 内容融合加入与 person 相同的元数据支路 | hybrid |
| `bm25-scope-reply` | 先范围融合，再从其前八条扩展同群回复；最多替换两条 | scope |
| `bm25-focus` | 问题有明确范围和背景从句时，加主问句召回，保留全文支路 | scope |

RRF 常数固定为 60，每个排名支路最多 50 条，最终给 QA 十条原始消息。内部候选数量随支路和去重变化，需要与最终 K 区分并报告。回复候选保留此前的一/两跳、相邻节点上限、正相关分数和两个替换位置约束。回复扩展后的 `score` 是扩展前的 RRF 分数，新增回复可能为零；解释最终选择时同时看 `reply_path` 和原始正文。

`I/my` 必须与个人状态谓词相连；显式用户 ID 和角色只在支持的主体句式中启用。背景中的 `driven by User_2` 和句尾接收者不自动成为目标。名称匹配只去掉同一位置被长名称完整覆盖的短名称；明确并列的范围保留。规则保守，不能理解所有自然语言指代。字段索引保持原 BM25 分词，带下划线的复合词不会自动做词形或别名扩展。

时间、人物、主问句、回复分别提供可解释的消融。先看全部 dev QA 和成对增损，再选择；小样本支持命中不能替代问答评测。Part 1/2 的数据和结果保持冻结。

## 完整开发集检索检查

本轮已获用户一次性最多 **8 GPU-hours** 的授权，包含检索预检、小样本 QA、八组完整 dev、冻结后的两组 test 及失败尝试。2026-11-21 在金融服务器 GPU 2、干净提交 `3cbf1dbac1a2f4f745125380307bf14c9d4b72c9` 上，完成八组 × 179 题的检索核对。检索阶段共用 0.1186 GPU-hours；E5 的配置、权重及逐题 Dense 前十与冻结对照一致。[完整检索证据](evidence/structured-retrieval-dev.json)

随后独立重算全部 1,432 个排序的 RRF 分数、稳定同分顺序、唯一 ID、回复路径和受保护前八条，全部通过。六组词法方法的 1,074 个完整轨迹与上一轮提交的证据完全一致。[独立核验](evidence/structured-retrieval-audit.json) · [上一轮 CPU 检索记录](evidence/structured-lexical-dev.json)

下表的支持命中表示至少包含一个已标注目标消息。18 道带目标的人工诊断题来自原有失败样本，选择有偏，不能当作整体召回率或问答准确率。BM25 的前十命中 7/18，正文 top-50 覆盖 10/18；Dense 前十命中 6/18。

| 方法 | 相对 BM25 改变前十成员 / 179 | 候选池支持命中 / 18 | 前十支持命中 / 18 | 新增 / 丢失 | 15 条 BM25 失败中的支持命中 | 最大候选池 |
|---|---:|---:|---:|---:|---:|---:|
| fields | 172 | 14 | 9 | 2 / 0 | 6 | 50 |
| scope | 99 | 13 | 8 | 1 / 0 | 5 | 100 |
| person | 104 | 13 | 9 | 2 / 0 | 6 | 100 |
| time | 104 | 13 | 9 | 2 / 0 | 6 | 100 |
| hybrid | 179 | 14 | 8 | 2 / 1 | 5 | 100 |
| hybrid-meta | 179 | 15 | 11 | 4 / 0 | 8 | 145 |
| scope-reply | 107 | 15 | 9 | 2 / 0 | 6 | 102 |
| focus | 99 | 13 | 9 | 2 / 0 | 6 | 135 |

范围/人物方法将 Manufacturing `user_implicit_3` 的目标从正文第 44 位提到第 1 位；角色信息将 Healthcare `multi_hop_48` 的目标从第 27 位提到第 3 位。focus 仅在 1/179 题触发；time 相对 person 改变 16 题，scope-reply 相对 scope 改变 14 题。对应的完整 QA 结果如下。

## 完整开发集与选型

2026-11-22 已完成八组各 179 题的真实回答及判分，共 1,432 条结果、2,864 次模型调用。全部运行无失败或不明确判分，逐题来源、固定划分、协议、检索结果及提示哈希通过核验。[完整 QA 证据](evidence/structured-qa-dev.json)

| 配置 | 正确 / 179 | 准确率 | 相对 BM25 新增 / 退化 | 本次 QA GPU-hours |
|---|---:|---:|---:|---:|
| BM25（归档对照） | 56 | 31.28% | — | — |
| Dense（归档对照） | 58 | 32.40% | — | — |
| fields | 55 | 30.73% | 20 / 21 | 0.461 |
| scope | 62 | 34.64% | 14 / 8 | 0.457 |
| person | 64 | 35.75% | 16 / 8 | 0.461 |
| time | 62 | 34.64% | 15 / 9 | 0.439 |
| hybrid | 62 | 34.64% | 23 / 17 | 0.486 |
| **hybrid-meta** | **65** | **36.31%** | **30 / 21** | **0.575** |
| scope-reply | 62 | 34.64% | 14 / 8 | 0.882 |
| focus | 63 | 35.20% | 15 / 8 | 0.882 |

按运行前的[选择规则](evidence/structured-selection-plan.json)，`hybrid-meta` 以正确数第一被选中，`hybrid` 是去元数据对照。[冻结记录](evidence/structured-freeze.json)于 **2026-11-21 20:11:20 UTC** 写入，早于任何 test 调用；2026-11-22 再用 dev 原始证据重放选型，除当前时间外全部字段一致。它相对 hybrid 新增 14 题、退化 11 题，净增 3 题；相对 person 新增 25、退化 24，说明总分接近的方法仍可能回答对不同问题。

并列时使用的次级标准是：已标注 F1–F6 的 BM25 失败中，新方法答对且目标进入前十的数量，然后是已测 GPU-hours，最后是固定配置顺序。hybrid-meta 有 4 条这样的修复，但样本有偏，不能代替全体指标。

## 冻结测试与当前边界

已完成并独立重算 `hybrid-meta` 的全部 **536 道 test**：177 题正确、无运行失败或不明确判分、5 次答案截断、0 次判分截断，1,072 次模型调用齐全。QA 用时 2.1964 GPU-hours。[测试审计及与归档基线的逐题比较](evidence/structured-selected-test.json)

| 方法 | 正确 / 536 | 准确率 | hybrid-meta 相对该方法新增 / 退化 |
|---|---:|---:|---:|
| BM25（原固定 test） | 178 | 33.21% | 56 / 57 |
| Dense（原固定 test） | 146 | 27.24% | 70 / 39 |
| hybrid-meta（冻结后运行） | 177 | 33.02% | — |
| hybrid（冻结后运行） | 未完成 | 暂不报告部分分数 | 等待补齐 |

**目前只能说 hybrid-meta 优于本地 Dense，整体与 BM25 持平；不能声称它在 test 上整体提升，也不能将相对 Dense 的收益归因于元数据。** 元数据自身贡献须等 hybrid 完整对照。对 BM25 的逐题 McNemar 精确双侧 p=1.0；对 Dense 为约 0.00385，但这是共享历史与问题族下的题级统计，不能推广为独立对话上的显著性。test 不再用于选型或调规则。

原 8 GPU-hours 在 hybrid 完成 246/536 题时触发硬时限，总用量为 **7.9945 GPU-hours**；中断原因是预算耗尽，退出码 124。[原始预算账本](evidence/structured-budget.jsonl)保留全部阶段，包含检索预检、失败尝试及外层启动时间。用户随后批准追加最多 **2 GPU-hours**，仅用于剩余 290 题、恢复及最终核验。2026-11-22 16:38（UTC+8）已启动等待空闲卡的独立续跑任务，启动时八张卡均无足够空闲显存；实时状态以服务器 `continuation-status.json` 为准，等待时间不计 GPU 预算。

硬中断留下 246 条完整记录和 493 条调用记录，其中最后一题只有已结束的 answer 调用。恢复工具保留原日志前缀、原配置和原账本，追加该题的失败 attempt，再让原运行器从 attempt 2 重试。原中断的 0.9632 GPU-hours 仍计入总成本；未落盘的 CPU 时间、GPU 峰值和索引耗时标为未知。重试必须使用相同答案提示哈希。完整验收预期为 536 条成功结果、537 次 answer、536 次 judge，以及 1 次历史中断，不把中断隐藏成干净重跑。

## 哪些方向保留，哪些停止

| 对比 | dev 新增 / 退化 | 判断 |
|---|---:|---|
| person / scope | 2 / 0 | 范围加明确人物或角色有可解释收益，保留简单消融 |
| time / person | 0 / 2 | 停止继续叠加“较新优先”；同阶段的新消息可能说的是另一项政策 |
| scope-reply / scope | 0 / 0 | 回复扩展改善部分支持命中，但没有增加判分正确数，停止扩大跳数或替换位置 |
| focus / scope | 1 / 0 | 只触发一道题，保留案例，不把单例规则包装为普遍提升 |
| fields / BM25 | 20 / 21 | 全文拼接元数据改变 172 题的前十成员且净退 1 题，停止继续投入 |
| hybrid-meta / hybrid | 14 / 11 | 完成冻结的主对照，报告正负例，等待 test 后判断贡献 |

当前最值得完善的是解释证据、人物识别与判分质量。范围和明确角色已是简单且可复用的机制；本轮没有依据再添加图模型、训练或更多启发式。如果以后研究相对日期归一化或回答时绑定消息来源，应作为所有方法共用的 QA 层单独实验，在新的验证协议下比较，不能在看过本轮 test 后继续针对它调参。

以下八例均来自 dev。主 agent 已将子 agent 的分析逐一与原问题、原消息、实际前十及输出核对；括号中的 ID 均须连同领域使用。

| 案例 | 实际证据与结果 | 对改进的含义 |
|---|---|---|
| Healthcare `knowledge_update_5` | hybrid-meta 将 `Msg_2947` 放到第 4；先锁核心字段、培训一周后再调其余字段的答案由错变对 | 新的支持证据确实补足旧计划 |
| Healthcare `multi_hop_48` | person 用明确 Health IT Specialist 找到 `Msg_24002`，从正文第 27 到最终第 3；答出 validate and freeze | 目标角色有用；提问者 Nurse Leader 不是目标人物 |
| Manufacturing `temporal_31` | `Msg_9004` 在 scope/person 中均第 1，后者才答对 2025-08-04 | 支持已存在，仍可能因其余上下文而答错，不能算新增召回 |
| Technology `knowledge_update_20` | time 挤掉第 8 的 `Msg_3723`，加入较新的 `Msg_10471`，正确答案变错 | 时间先后不能代替政策与议题的匹配 |
| Technology `multi_hop_31` | 两种 hybrid 均将 `Msg_17842` 放第 1；meta 将 7 月 22 日误答成另一任务的 7 月 28 日 | 同群中多个相对日期仍会互相干扰 |
| Technology `temporal_2` | meta 新增第 10 的 `Msg_6810`（7 月 23 日的 Friday），却回答后续消息的 7 月 27 日 | 找到正确消息仍不足以区分原截止日与后续催办 |
| Technology `multi_hop_2` | 回复路径 `638 → 832 → 847` 把 `Msg_847` 放第 10，判分仍错 | 需要复核 judge，见下文；不能仅按分数认定回复没有作用 |
| Technology `knowledge_update_30` | focus 将 `Msg_4523` 放第 3，答对四个冲突字段及重读快照后重试 | 主问句有用的单例，全 dev 仅此一次触发 |

`multi_hop_2` 的参考答案是 “A single fallback rule.”。源消息同时要求 stale availability 使用一个 fallback、hard blocks 使用另一条更严格规则。扩展后的回答复述这一区别；Qwen judge 却以“出现两条规则”为由判错。这是**可能的判分假阴性，待队友人工复核**。本轮保留原判分和所有统计，不擅自改成正确，也不把一次人工疑问用来调整整个评分口径。

成本也须区分算法与运行环境。scope-reply/focus 的检索总耗时约 30.5/31.1 秒，与 scope 的 29.2 秒接近；但后两组回答及判分的实测吞吐约 15.1 tokens/s，早期 scope 约 29.4。它们约翻倍的 GPU-hours 主要发生在生成阶段，不能据此声称回复检索本身慢一倍。现有日志没有足够证据确定生成变慢的外部原因。

## 核验与复现

八组先各做两个 dev 问题的真实预检：16 次回答、16 次判分全部成功，没有空输出、解析失败或截断，约 0.0610 GPU-hours，不用于选型。[预检回执](evidence/structured-preflight.json)保留配置与核验结果。原两个回复方法的历史 dev 结果保留作对照，不与新方法混成一次实验。

本轮恢复只改离线审计工具，检索、QA、模型和冻结源码均保持不变。针对中断前缀保留、重复调用、缺失收据、Unicode 文本、重试提示和资源缺失的 25 项定向测试通过；最终完整 CPU 套件 **119 项通过**。Part 1/2 的 1,430 条记录、24 道分析题、38 条失败标注及 61 条源消息重新核验通过。

子 agent `gpt-6.2-sol`（high）只读审查了 dev 取舍及恢复逻辑；主 agent 核验八个源案例、修复其发现的 Unicode 读取与计时范围问题，并运行测试。该审查不是独立人工标注；运行时没有提供 agent token 统计。

## 检索检查

提交源码并保持工作区干净后，在根目录运行：

```bash
python scripts/explore_retrieval.py --output /tmp/structured-lexical-dev.json
```

默认覆盖六个无需新模型调用的方法、四域全部 179 道 dev。脚本验证上游提交、输入文本、固定划分和归档 BM25 的逐题 ID/分数；Windows 换行差异会单独记录。每题保留实际候选列表、最终 ID、融合分数和回复路径；检索结束后才加载人工支持标注。正式证据需要 `project_git.status` 为空，调试输出清理后不作为正式结果。

完整矩阵的检索核对使用已有 E5，复现命令为：

```bash
python scripts/explore_retrieval.py \
  --methods bm25-fields bm25-scope bm25-person bm25-time hybrid hybrid-meta bm25-scope-reply bm25-focus \
  --embedding-model "$E5_MODEL" --embedding-revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --device cuda --output /tmp/structured-retrieval-dev.json
```

脚本核验冻结的 E5 配置和权重，并核对每题 Dense 前十与归档控制一致。新增元数据模块纳入实现签名，后续分片不能混用不同实现。

## 问答、比较与冻结

先对每个配置做少量真实 dev 问答，检查提示和资源；随后逐方法运行全部 179 题。复用同一 Qwen3-8B、E5-small-v2、seed 42、回答 512 tokens、判分 256 tokens。每个正式 run 只放一个方法，让其 GPU-hours 可以独立归属；不同方法共用原运行入口，不引入新的生成框架。

```bash
# METHOD 取上表中的一个名字；RUN 是新结果目录。模型权重使用服务器已有路径。
python scripts/run_local_baselines.py --methods "$METHOD" --full --partition dev \
  --domains Finance Technology Healthcare Manufacturing \
  --model "$QWEN_MODEL" --embedding-model "$E5_MODEL" \
  --model-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --embedding-revision ffb93f3bd4047442299a41ebb6fa998a38507c52 --output "$RUN"

python scripts/analyze_exploration.py --runs /path/to/run1 /path/to/run2 \
  --probe /path/to/structured-retrieval-dev.json --bm25-audit /path/to/part1-bm25/generation-audit.jsonl
```

汇总检查数据、模型、源码、候选、调用和逐题完整性。提示哈希检查其相对 BM25 的变化是否与上下文/答案变化一致，同时锁定生成代码和输入；它不是重新加载 tokenizer 后逐字重建提示的独立核验。输出包括整体/六类/四域、每对方法修复与退化、截断及成本。`total_qa_gpu_hours` 只合计正式 QA，CUDA 检索和预检另外计入本轮预算。未提供全部八组时 `complete_matrix=false`，不会把部分实验标成完整。原两个回复方法的问答控制也纳入比较。

dev 比较后已冻结 hybrid-meta 与 hybrid，各评估 536 道 test。测试结果用于评价与报告，不再用于选规则。追加预算只补齐中断的同一比较；没有启动新的候选或调参。


自动选型及冻结后的离线核验使用以下入口。它们只处理结果，不调用模型。已有冻结记录不可覆盖；续跑仅执行最后的比较审核。

```bash
python scripts/select_exploration.py --comparison "$ROUND/dev-comparison.json" \
  --probe "$ROUND/retrieval-dev.json" --plan "$ROUND/selection-plan.json" --output "$ROUND/freeze.json"

python scripts/analyze_frozen_test.py --selected-run "$SELECTED_TEST_RUN" --control-run "$CONTROL_TEST_RUN" \
  --freeze "$ROUND/freeze.json" --dev-comparison "$ROUND/dev-comparison.json" \
  --selection-plan "$ROUND/selection-plan.json"
```

本轮服务器目录为 `/home/ai4s_group/XZG/NLP/sgmem-part3/results/runs/metadata-round-20261121`。原 dev/收尾会话已结束。**不要重启旧的 `finish-round.py`，也不要在续跑结束前切换服务器源码提交。** QA 仍从干净的 `3cbf1db` 运行，修复后的审核工具单独放在运行目录 `tools/`。

补额任务为 tmux `nlp-frozen-test-resume`，入口 `continue-frozen-test.py`。它核验冻结源码与恢复记录，等无计算进程且至少 22,000 MiB 空闲的一张卡，复查后执行原命令加 `--resume`。单次超时为 7,180 秒，另留终止余量，总上限 2 GPU-hours；失败即停，不循环消耗预算。`continuation-plan.json`、`continuation-budget.jsonl`、`continuation-status.json`、`test-hybrid-resume.log` 记录授权、实际用量和进度；原 `budget.jsonl` 不再追加。

完成后自动审核两组 test，写 `test-comparison.json`，再将续跑状态改为 `complete`。状态只有到这一步才算本轮比较完成。正式运行目录中的原启动配置、日志和原始结果保留用于追溯。

一般的同配置中断可直接使用 `--resume`。本次存在已完成调用但尚未写入问题检查点的硬中断，先使用下面的一次性恢复入口。默认只查看，不修改原日志；`--apply` 只可执行一次，且必须确认原进程已经结束。它严格限于首段会话被预算超时结束的情形，不是通用日志修补器。

```bash
python scripts/recover_interrupted_run.py --run "$ROUND/test-hybrid" --ledger "$ROUND/budget.jsonl"
# 核对原始记录与预算后才加 --apply；本轮已经执行，不要再次应用。
```
