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

范围/人物方法将 Manufacturing `user_implicit_3` 的目标从正文第 44 位提到第 1 位；角色信息将 Healthcare `multi_hop_48` 的目标从第 27 位提到第 3 位。focus 仅在 1/179 题触发；time 相对 person 改变 16 题，scope-reply 相对 scope 改变 14 题。它们的 QA 收益仍须以完整 dev 判断。

## 本轮后台运行状态

八组方法分别通过两个开发问题的小规模问答检查：16 次回答、16 次判分全部成功，没有空输出、解析失败或截断。预检占用约 0.0610 GPU-hours，不用于选型。[预检回执](evidence/structured-preflight.json)保留了配置与核验结果，临时原始输出已清理。2026-11-22 凌晨，首个完整方法 fields 已核验：55/179（30.73%），相对 BM25 的 56/179 修复 20 题、退化 21 题，完整 QA 占用约 0.4607 GPU-hours。它的支持命中增加没有转化为整体问答提升。其余配置正在运行，尚无完整矩阵或最终 test 结论。

用户随后要求配置自动收尾，次日再分析。服务器的 `results/runs/metadata-round-20261121/` 保存本轮配置与轨迹。后台流程依次执行：

1. 等待八组 dev 全部成功，审核固定 179 题、来源、调用、提示与检索记录。
2. 依照[运行前的选择规则](evidence/structured-selection-plan.json)选型，写入一次性的 `freeze.json`。纯 hybrid 是内容对照；两种旧回复方法可作为元数据方法的回退候选。
3. 冻结后为所选方法和去元数据对照分别重新运行 536 道 test。若选 hybrid-meta，对照为 hybrid；其他选择的对照为 BM25。
4. 审核完整测试记录，生成 `test-comparison.json` 和最终 `status.json`。错误、缺失记录、源代码变化或预算耗尽会停止后续执行，不自动绕过。

开发集并列时，先比较已标注 F1–F6 的 BM25 失败中，新方法答对且目标消息实际进入前十的数量，再比较已测 GPU-hours，完全相同时使用固定配置顺序。这是有偏诊断样本上的次级标准，不改变以完整 dev 正确数为首要标准的约定。所有 GPU 阶段串行累计计时，`gpu_stage.py` 对实际子进程执行累计 8 小时的硬时限。后台不使用 test 反馈调整选择，也不自动提交未经分析的最终结果。

新增的选型与测试审核工具由子 agent 实现并由主 agent 审核；主 agent 补充每题十条唯一原始消息的检查后，连同自动选型的边界验证，22 项针对性合成测试通过。服务器还用首组真实结果验证了不完整矩阵会在进入 test 前被拒绝。子 agent 使用运行时可用的 `gpt-6.2-sol`，high；任务仅涉及离线审核，没有模型调用。测试包括丢失/重复记录、协议变化、重试、冻结时间及上下文格式。未提供 agent token 用量。

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

完成 dev 比较后冻结最优方案与对应去元数据对照，各跑一次 536 道 test。测试结果用于评价与报告，不再用于选规则。本轮已获完整预算授权；在后台流程完成并核验结果前，不将实现或检索命中视作最终问答提升。


自动选型及冻结后的离线核验使用以下入口。它们只处理结果，不调用模型；本轮服务器的后台任务已将这些步骤连接好。

```bash
python scripts/select_exploration.py --comparison "$ROUND/dev-comparison.json" \
  --probe "$ROUND/retrieval-dev.json" --plan "$ROUND/selection-plan.json" --output "$ROUND/freeze.json"

python scripts/analyze_frozen_test.py --selected-run "$SELECTED_TEST_RUN" --control-run "$CONTROL_TEST_RUN" \
  --freeze "$ROUND/freeze.json" --dev-comparison "$ROUND/dev-comparison.json" \
  --selection-plan "$ROUND/selection-plan.json"
```

本轮服务器目录为 `/home/ai4s_group/XZG/NLP/sgmem-part3/results/runs/metadata-round-20261121`。`nlp-metadata-dev` 会话运行八组 dev，`nlp-metadata-finish` 会话等待后续条件并自动收尾。当前检索与 QA 固定运行于 `3cbf1db`；收尾工具保存在该运行目录的 `tools/` 中，避免中途改变正在运行的源码。`finish-round.py`、`finish-launch.sh`、`launch-dev.sh` 和 `gpu_stage.py` 是本次运行的启动配置，保留在正式运行目录中。明天从 `status.json`、`budget.jsonl`、`freeze.json` 和 `test-comparison.json` 接续；后台不会替代最终案例分析和课程报告。
