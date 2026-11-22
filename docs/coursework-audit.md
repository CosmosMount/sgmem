# Topic 12 要求与评分审核

审核日期：2026-11-22。实验交付基线为 GitHub 分支 `codex/part3-bounded-replies` 的 `99f9aba`；本次只完善复现入口和说明，不修改冻结方法、模型调用、正式标签或结果。

**判断：Part 1、2、3 的核心实现、实验和分析已实质完成，足以进入最终交付。停止本轮方法搜索；下一步优先完成约 6 页报告、海报/演示和个人代码面试准备。** 完成实验不等于完成课程提交，也不保证高分。当前最强结论是元数据帮助 hybrid 恢复部分错误，整体未超过 BM25。

## 原文与完成证据

本次重新读取课程原件，并渲染检查评分表；不是沿用旧 syllabus 推断要求。

- `AIAA4051_26Fall_Final_Research_Projects.pdf` 第 44–46 页；SHA256 `709b38ae78f94e51ec362257eb1d03e9f97de8baf2613e09d24351a852e69028`。允许本地 Qwen/E5，EverMemBench 是可选扩展，明确接受经过充分分析的负结果。
- `AIAA4051_Project_Evaluation.pdf` 第 1 页；SHA256 `bafaa36df94696c4bf5f0d0d315b4262f76ee5707ba80b83a0c4360057a8f2de`。海报、报告、个人代码面试分别占课程总评 10%、13%、12%，合计 35%。不要将此比例与 Topic 12 的任务分配 20/40/30/10 混淆。

| 任务 | 完成证据 | 必须保留的边界 |
|---|---|---|
| Part 1（20%）：BM25、Dense、官方流程、总体及六类准确率 | 两法各 715 题，四域完整语料；固定 dev/test 179/536；上游提示、结果解析和 BM25 排序对齐；[结果和资源](part12-full-results.md) | 使用本地 Qwen3-8B/E5-small-v2，是允许的本地适配，不是 GPT-5/商业 embedding 原模型配置复现 |
| Part 2（40%）：错误分类、分布、代表案例及元数据能否找回证据 | 固定 dev 配额样本 24 题、38 条失败记录、九类标签、61 条可核查源消息；speaker/group/time/reply 均有诊断 | AI-assisted source review，非独立人工金标；样本分布不能当作所有失败的总体分布；F9 判分疑点不混同方法错误 |
| Part 3（30%）：一个聚焦修改、元数据参与选择、同条件对照、独立 dev/test | 八组完整 dev；预先冻结 hybrid-meta 与 hybrid，各 536 题 test；与 BM25/Dense 的总体和六类结果、消融、增损与案例齐全；[完整证据](part3-exploration.md) | 最终 K=10 和 QA 设置一致；内部候选列表与计算量不同，须披露。元数据不是仅在展示阶段附加 |
| Part 4（10%）：约 6 页报告、可运行代码、代表性轨迹/演示 | 代码、配置来源和轨迹已有，仓库内可复核 | 仓库内长篇实验记录不是最终 6 页报告；尚需面向课程评审整合 |

本次再次执行 `verify_part12_evidence.py --benchmark external/GroupMemBench`，1,430 条结果、划分、38 条标注及 61 条源消息通过。使用 Python 3.11 对原始 test 两臂重跑 `analyze_frozen_test.py`，与归档报告逐字段一致；现有 Python 3.13 环境的完整 CPU 套件 **119 项通过**。本次没有改变算法代码，无新增推理或判分。

## 最影响得分的收尾工作

| 优先级 | 收尾事项 | 对应评分与验收方式 |
|---|---|---|
| 必做 | 写成约 6 页完整报告，主线为“失败诊断 → 简单元数据支路 → 受控比较 → 有限收益及代价” | 报告的问题动机 8、相关工作 8、方法 15、实验 23、分析 15、写作引用 8 分。每张表可追溯到固定 JSON 和脚本；文献应说明定位，不能把 RRF 本身作为新算法 |
| 必做 | 单列质量与计算成本，说明资源测量边界 | 报告效率共 23 分：资源 8、效率选择 8、质量/成本 7。使用已有 latency、tokens、索引时间、GPU-hours 和峰值显存；未知字段明确写缺失，不用额外跑大实验补漂亮数字 |
| 必做 | 准备海报及短演示，展示机制、基线/消融、正负案例 | 海报基础内容 60、参与 10、进阶 30；进阶包含 baselines/ablations、theory、case studies 各 10 分。讲清 RRF 公式、复杂度及假设即可支撑方法解释，不捏造定理或保证得分 |
| 必做 | 每位成员能讲解核心实现、追踪一个结果、解释一次失败 | 个人代码面试：理解 60、可读性 40。围绕下表演练；无需再加复杂模块 |
| 建议 | 队友复核少量 AI 辅助案例与 judge 疑点，记录独立意见 | 优先四条 F9、一个明确角色修复、一个群组/主题修复及测试中的日期判分不一致。老师原文未强制人工标注；这是提高可信度的建议。保留正式模型评分，不静默改分 |
| 交付前 | 集成团队分支、确认可获取的模型/原始结果路径，做一次新机器小样本验收 | 当前成果在 `codex/part3-bounded-replies`；`main` 尚未集成 Part 3。迁移时使用正式依赖和固定 revisions，不能把旧 Docker 单题验证等同于目标机器复现成功 |

**论文主表使用同一 536 题 test 分母：** BM25 178/536（33.21%）、Dense 146/536（27.24%）、hybrid 164/536（30.60%）、hybrid-meta 177/536（33.02%）。不能把 Part 1 的 715 题全量准确率直接与 Part 3 的 536 题 test 比较。meta 相对 hybrid 新增 35、退化 22，净增 13，题级 McNemar p=0.111；相对 BM25 少 1 题。当前证据不足以声称显著或稳定超越最强基线。

## 把结构性失败与方法效果连起来

这一步整理已有记录即可，无需新评分。源标签见 [Part 2 annotations](evidence/part12-full-20261105/failure-annotations.jsonl)，排序见 [dev 检索轨迹](evidence/structured-retrieval-dev.json)，回答见 [dev QA](evidence/structured-qa-dev.json)。

| 证据链 | 原诊断与实际变化 | 可以支持的结论 |
|---|---|---|
| Healthcare/dev `multi_hop_48` | 原 BM25 混淆目标角色，Dense 跨项目混淆。问题明确指 Health IT Specialist；`Msg_24002` 在 scope 前十缺失，person 将其置于第 3 并由错变对；hybrid 缺失，hybrid-meta 置于第 4 并答对 validate and freeze | 明确角色参与候选选择能修复一个已识别的结构性失败；提问者不是默认目标人物 |
| Healthcare/dev `knowledge_update_5` | 原 Dense 跨群组混合字段操作。Hand Hygiene 主题支路使 `Msg_2947` 从 hybrid 前十外进入 meta 第 4，回答恢复“先锁核心字段，培训后再调整其余字段” | 范围选择能找回正确更新；不是所有“知识更新”题都靠时间排序解决 |
| Technology/test `multi_hop_24` / `term_ambiguity_30` | 前者新增所需 `Msg_27529` 并答对；后者挤掉正确 `Msg_1010` 后弃答 | 冻结方法既有修复也有排挤证据的代价；不能只展现成功案例 |

以上是定性机制证据。六类 QA 分数不是 speaker/group 错误发生率；这些案例也不是穷尽标注，不能据此宣称已测得全 test 的结构性错误下降比例。支持已召回但回答仍错误的案例应保留，避免把 QA/judge 局限全部归因于检索。

## 方法与成本如何讲清楚

最终方法是内容 BM25 与 E5 排名，加上一条由公开名称/明确目标人物限定的词法排名，再做等权 RRF：对各列表中出现的消息累加 `1 / (60 + rank)`，同分保留原始索引次序，最终取十条原始消息。没有训练新模型，也没有额外 LLM 路由调用。

每个支路最多 50 条；hybrid 两路最多 100 条，hybrid-meta 三路理论最多 150 条，开发集实际最大 145 条。若列表数为 L、深度为 d、去重候选数为 C，融合与排序约为 O(Ld + C log C)，不含前面的全库 BM25/E5 检索。现有实现使用全量打分/排序；不能把这条融合复杂度误写成整个系统复杂度。报告要区分相同的最终检索上下文预算与不同的内部候选计算量；当前对照测的是“加入元数据候选支路”的整体影响，未隔离固定候选池上的纯重排效应。

可直接追溯的非重叠成本为：Part 1 全量 3.5907 GPU-hours、两轮已归档回复 dev 0.4303/0.4376、本轮完整探索及续跑 8.7841，合计 **13.2427 GPU-hours**。这是已列出运行的合计，不包含所有历史 pilot，不能作为整个项目总用量；报告中补齐旧小样本账目或明确这一边界。八组 dev 和两组 test 的成本已在 8.7841 中，不能再次相加。shared GPU 时段的吞吐变化、一次硬中断缺失的资源字段均已在[原报告](part3-exploration.md)披露。

## 代码面试与复现入口

| 应能回答的问题 | 最短代码/证据路径 |
|---|---|
| 数据如何读取？为何 dev/test 不重叠但仍共享历史？ | [`data.py`](../src/sgmem/data.py)、[`shards.py`](../src/sgmem/shards.py)、固定 [`split.json`](evidence/part12-full-20261105/split.json) |
| BM25、Dense 与元数据方法到底差在哪里？为何不能默认按 asker 筛？ | [`retrieval.py`](../src/sgmem/retrieval.py)、[`structured.py`](../src/sgmem/structured.py)、[`query_metadata.py`](../src/sgmem/query_metadata.py) |
| 参考答案在哪里使用？如何保持相同 QA/judge 口径？ | [`qa.py`](../src/sgmem/qa.py)、[`runner.py`](../src/sgmem/runner.py)；gold 只用于判分/事后分析 |
| 如何从 177/536 追到具体问题和上下文？ | [`structured-test-comparison.json`](evidence/structured-test-comparison.json) → [`structured-test-cases.json`](evidence/structured-test-cases.json) 或服务器 raw predictions；[`analyze_frozen_test.py`](../scripts/analyze_frozen_test.py) 重算 |
| 为什么回复支持命中改善而 QA 没有提高？为什么不能继续挑 test 最好方案？ | [Part 3 正负案例与冻结选择](part3-exploration.md)、[`structured-freeze.json`](evidence/structured-freeze.json) |

本次补充 [`requirements-local.txt`](../requirements-local.txt)，并明确它与 HTTP/Docker 环境的区别。已核验服务器实际包版本、导入以及 `pip check`；主要五个包与 Part 1/Part 3 清单一致。没有为此升级服务器或重新安装依赖。新机器应先验证环境和小样本，再考虑正式重跑。

## 停止线

不建议继续扩大回复跳数、增加时间/人物启发式、训练图模型、更换大模型、增加第二基准或反复寻找更好 test 分数。当前数据不支持这些投入能稳定换来更高课程分数。只有发现影响公平性/正确性的具体错误，或教师新增明确要求，才值得重开实验；否则把时间用于上述最终交付。

本次由 `gpt-6.2-sol`（high）子 agent 只读复核 Part 2/3，主 agent 重新核对原文、原始统计及两条结构修复的标签/排序/回答后接受结论。此复核不属于独立人工标注；未新增模型调用或 GPU 实验。
