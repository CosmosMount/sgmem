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

## 已完成的完整开发集检索检查

2026-11-21 在金融服务器的干净提交 `7b767a1a136e066d9db3058bb18e8c280ba0f5d8` 上完成六组 × 179 题，共 1,074 次检索，耗时 216.11 秒。未调用 Qwen、judge 或新的 E5 推理。全部最终结果为十个唯一消息 ID；每道 BM25 对照的 ID、顺序和分数均与归档一致。正式逐题候选和统计见 [structured-lexical-dev.json](evidence/structured-lexical-dev.json)。发布前还从保存的候选列表独立重算全部 1,074 个排序，核验分数、受保护的前八条、回复路径及候选预算，全部一致。

下表的支持命中指至少包含一个已标注目标消息。18 道带目标的人工诊断题来自原有失败分析样本，选择过程有偏，不能把命中比例理解为整体召回率或 QA 准确率。BM25 在这 18 题的前十命中 7 题；若将正文候选扩至 50 条则覆盖 10 题。

| 方法 | 相对 BM25 改变前十成员的题数 / 179 | 候选池支持命中 / 18 | 前十支持命中 / 18 | 相对 BM25 新增 / 丢失 | 15 条 BM25 失败中的前十命中 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0 | 10（正文 top-50） | 7 | 0 / 0 | 4 |
| fields | 172 | 14 | 9 | 2 / 0 | 6 |
| scope | 99 | 13 | 8 | 1 / 0 | 5 |
| person | 104 | 13 | 9 | 2 / 0 | 6 |
| time | 104 | 13 | 9 | 2 / 0 | 6 |
| scope-reply | 107 | 15 | 9 | 2 / 0 | 6 |
| focus | 99 | 13 | 9 | 2 / 0 | 6 |

六个方法的实际最大候选池分别为 50、100、100、100、102、135，最终 K 均为 10。两组 Hybrid 已实现并覆盖离线接口测试，真实 E5 检索待整轮模型授权。

这一轮明确了四种不同机制：

- **范围和人物**：制造业 `user_implicit_3` 的目标从 BM25 第 44 位进入范围/人物方法第 1 位；医疗 `multi_hop_48` 由角色信息把原第 27 位支持提到第 3 位。两者都使用公开查询和消息字段。
- **主问句**：只在 1/179 题触发，Technology `knowledge_update_30` 的支持进入第 3 位。覆盖面小，尚无泛化证据，应与 scope 独立比较。
- **回复组合**：相对 scope 改变 14 题，Technology `multi_hop_2` 的支持进入第 10 位。候选池能覆盖 15/18，但最后仍只有 9/18，说明后续选择与 QA 都可能成为瓶颈。
- **时间**：相对 person 改变 16 题，支持命中没有额外增加。仍需看真实回答是否受益或受损，不能先宣称这个方向有效。

这些方法补回的病例不同。首轮完整 QA 前保留八组矩阵，不依据 18 道诊断题挑定赢家，也不继续扫权重。Windows 和服务器均为 **89 项测试通过**；Part 1/2 的 1,430 条结果、24 道样本、38 条标注及 61 条源消息复核通过；原两个 179 题回复方法的 QA 报告逐项重算一致。测试覆盖合成边界和接入流程，新的 Qwen 推理仍需要小规模真实预检。

截至本次提交，尚未完成新八组的 dev QA 和冻结 test。已申请一次性最多 8 GPU-hours，等待用户确认；最后检查时服务器八张卡均不足 6 GiB 空余显存，不满足现有 Qwen 运行所需的约 20 GiB 空余量。正式结果报告不能把本节写成已完成 Part 3 的最终问答验证。

## 检索检查

提交源码并保持工作区干净后，在根目录运行：

```bash
python scripts/explore_retrieval.py --output /tmp/structured-lexical-dev.json
```

默认覆盖六个无需新模型调用的方法、四域全部 179 道 dev。脚本验证上游提交、输入文本、固定划分和归档 BM25 的逐题 ID/分数；Windows 换行差异会单独记录。每题保留实际候选列表、最终 ID、融合分数和回复路径；检索结束后才加载人工支持标注。正式证据需要 `project_git.status` 为空，调试输出清理后不作为正式结果。

得到整轮模型预算授权并确认 GPU 后，使用已有 E5 完成完整矩阵的检索检查：

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

完成 dev 比较后冻结最优方案与对应去元数据对照，各跑一次 536 道 test。测试结果用于评价与报告，不再用于选规则。新增 QA 需要用户确认整轮预算；当前这份实现与 CPU 检查不意味着新方法已获得更高问答准确率。
