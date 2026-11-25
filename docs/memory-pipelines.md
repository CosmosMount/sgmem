# Dense X、LightMem、RAPTOR、HippoRAG 2 完整探索

更新于 2026-11-25。范围由用户确认：同学原表第 2–5 项。IRCoT 不在本轮范围。目标是完成四种方法的核心流程、相同口径的效果比较、机制消融、成本和失败分析；保持简单代码，不把四个上游框架混装进正式 QA 环境。

从 GitHub `codex/part3-bounded-replies@8de9d301ed765425ef851ee031b9c716dcc42c58` 创建堆叠分支 `codex/memory-pipelines`。开始时 `origin/main` 仍为 `c32a47c6527a646229ddbd7065fed0e6b4c6bee1`，没有更新。新分支依赖尚未合并的 Part 3 代码，不表示已合入 main。

## 当前状态

| 工作 | 状态 | 完成判据 |
| --- | --- | --- |
| 四篇论文、官方源码及依赖核查 | 完成；父 agent 复核关键源码 | 固定来源，明确论文核心与适配偏差 |
| 公共数据导出 | 完成离线核验 | 120,000 条消息、179 道 dev，标签隔离，来源 ID 保留 |
| Dense X 提取与原文回溯 | 四域 120,000 条完整建库；五个条件各完成 179 题 dev 问答与判分 | 基础/元数据配对已冻结，536 题 test 尚未启动 |
| RAPTOR | 四域 120,000 条完整建树；五个条件各完成 179 题 dev 问答与判分 | 本轮未选入 test；开发集结果见下表 |
| LightMem | 真实小测通过；Healthcare 10 个群组中 2 个完成，覆盖 5,859 条源消息 | 其余 8 个失败，尚无完整领域 QA 结果 |
| HippoRAG 2 | 真实小测通过；Healthcare 30,000 条源消息建图完成 | 重新加载索引的配置校验失败，正式检索和 QA 尚未完成 |
| 检索结果与固定 QA/judge 接入 | 已接入全部计划条件；此前 286 项离线回归与 6 项真实模型接口检查通过；现有 10 组 dev 共 1,790 条有效评分 | LightMem/HippoRAG 2 的正式 QA 仍待运行 |
| GPU 调度 | 本次续跑前已用 18.1687 GPU-hours；之前控制程序因 SSH 断开停止 | 本轮累计不超过 30 GPU-hours，失败和服务加载也计入 |

已有 Part 1–3 证据保留。本轮的效果比较分阶段记录。有合适卡时可直接运行，本轮最多 30 GPU-hours。可核对的历史不重叠资源小计为 13.3489 GPU-hours，另有少量历史计时缺失；本轮与历史累计仍须满足课程小于 50 GPU-hours 的约束。

截至本次小测完成，本轮已记录 **1.096281 GPU-hours**，包括服务加载、失败尝试和占卡期间的 CPU 工作。最后一批的总退出码仍为 1：最初的探针错误地要求原始模型关系零重复，随后独立修正探针和完整 HippoRAG 小测通过；没有覆盖失败账目。各方法最后小测以及 6 项原生/原文接口检查均有独立完成记录。紧凑证据与远端原始收据哈希见 [小测核验记录](evidence/memory-pilot-20261125.json)。

其后完成了课程服务器 A40 上的整卡成本试跑和 Dense X 全量构建。整卡试跑共 0.5770 GPU-hours，已包括加载、兼容预检和下表各阶段。Dense X 全量使用相同服务、每批 256 条输入和最多 32 并发，实测 3.6125 GPU-hours。初期小测和上述两次执行合计 5.2858 GPU-hours；后续阶段按两台服务器的实际账本另计。

| A40 成本试跑阶段 | 源消息数 | 耗时 | 实际模型调用 |
| --- | ---: | ---: | ---: |
| Dense X | 200 | 77.69 秒 | 200 |
| LightMem | 1,000（四域各 250） | 1,355.79 秒 | 754 |
| RAPTOR | 1,000 | 114.11 秒 | 262 |
| HippoRAG 2 | 1,000 | 371.74 秒 | 1,998 |

逐行核验了 Dense X 的 120,000 条源 ID、源哈希及回退正文。最终有 500,286 个检索单元，503 条源消息回退原文（0.419%）；其中 1 条达到生成上限。四种 Dense X 排名及分句对照均覆盖冻结的 179 道 dev。整卡试跑、全量建库和 RAPTOR 截断复测见 [扩容核验记录](evidence/memory-scale-20261125.json)。

## 完整开发集对照（2026-11-25 晚间核验）

下表均使用固定的 179 道 dev、Qwen3-8B QA/judge 与 E5-small-v2。主比较从候选回溯原消息；原生上下文和原文预算条件单列，不与主比较混为同一种上下文。核对十份预测文件与冻结审计记录的哈希后，所有条件均有 179 条有效评分，没有运行错误或无法判分的题。少数回答达到生成上限，其数量如下。按领域、类别统计、逐题得失、运行版本及结果哈希保存在[开发集核验记录](evidence/memory-dev-20261125.json)。

| 条件 | 正确数 / 179 | 准确率 | 达到回答长度上限 |
| --- | ---: | ---: | ---: |
| Dense X：命题检索后回溯原文 | 52 | 29.05% | 1 |
| Dense X + 元数据 | 55 | 30.73% | 3 |
| Dense X：分句对照 | 54 | 30.17% | 0 |
| Dense X：直接使用命题上下文 | 36 | 20.11% | 1 |
| Dense X：原文预算对照 | 54 | 30.17% | 5 |
| RAPTOR：多层检索后回溯原文 | 48 | 26.82% | 1 |
| RAPTOR + 元数据 | 54 | 30.17% | 1 |
| RAPTOR：仅检索叶子 | 56 | 31.28% | 4 |
| RAPTOR：原生上下文 | 43 | 24.02% | 3 |
| RAPTOR：原文预算对照 | 44 | 24.58% | 4 |

相同开发题的归档对照为 BM25 56/179、消息级 Dense 58/179、hybrid 62/179、hybrid-meta 65/179。本轮已完成方法尚未超过 hybrid-meta。

- Dense X 加元数据后新答对 12 题、丢失 9 题，净增 3；RAPTOR 加元数据后新答对 7 题、丢失 1 题，净增 6。小样本增益不等同于稳定的总体提升。
- Dense X 基础版比简单分句少答对 2 题；RAPTOR 基础版比仅叶子少答对 8 题。当前设置未证明额外的生成和层级结构能带来整体收益。
- 直接使用 Dense X 命题比原文预算对照少答对 18 题，RAPTOR 原生上下文少答对 1 题。需要结合逐题证据分析信息损失，不能仅以总分判定具体原因。

正式逐题输出保存在金融服务器本轮目录的 `densex-dev-benchmark/` 和 `raptor-dev-benchmark/`；共同核验记录为 `finish-study-20261125/dev-audit.json`。代码、数据划分、模型版本和上下文差异均可从上述精简证据追溯。这些是本项目适配实验，不是原论文设置的精确复现。

### 未完成项与冻结边界

LightMem 的 Healthcare 群组 01、09 完成；其余八个失败。至少两个失败请求的输入加预留输出超过服务的 8,192-token 上限，其他失败记录为生成未正常结束或缺少 usage。完整索引尚未合并，不能报告该方法正式准确率。

HippoRAG 2 的 Healthcare 图已保存，含 29,932 个唯一 passage、51,460 个 phrase 节点和 103,481 条唯一抽取三元组。图检索及 passage dense 对照均在重新加载索引时遇到 `Index schema/config mismatch`，没有完整 QA 分数。缓存和图仍保留；该故障属于集成问题，不能据此评价方法效果。

2026-11-25 07:00:56 UTC，按既定规则在已有完整 dev 的 Dense X、RAPTOR 两个候选中冻结 `densex` / `densex-meta`。该选择不代表四种方法已全部比较。后续 test 发布步骤因 SSH 连接断开停止，两组 536 题 test 当时均未启动。续跑保留原冻结记录，不重选方法或修改参数；LightMem/HippoRAG 2 的后续分析仍限 dev。

晚间恢复核验中，LightMem 已通过 4,980-token 长输入请求和 40 条真实消息的完整小测（包括 15 次 sleep 调用）；HippoRAG 已重开原图并完成 Healthcare 的 33 题 passage-dense 导出，随后按固定上游的候选映射方式修正识别阶段的非精确匹配。291 项离线测试通过。小测细节见[恢复核验记录](evidence/memory-recovery-20261125.json)，后续正式结果单独验收；这些核验不计入 benchmark 准确率。

## Dense X 开发集案例与重放核验

基础 Dense X 对 179 道题的 10 条原消息回溯评测得到 **52/179（29.05%）**。179 道题均有有效判分，1 个回答达到生成长度上限。使用原始排列、固定 Qwen tokenizer 和保存的生成记录重放了 358 个回答与判分提示，哈希逐项一致。

| 同题对照 | 对照正确数 | Dense X 新答对 | Dense X 丢失 | 净变化 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 56/179 | 21 | 25 | −4 |
| 消息级 E5 Dense | 58/179 | 25 | 31 | −6 |

这项比较当前未显示整体收益。Healthcare `knowledge_update_5` 中，Dense X 补回了 `Msg_2947` 并答对；Finance `temporal_21` 中，Dense X 的前 10 条缺少 Dense 曾取回的 `Msg_8383`，回答漏掉了冻结日期。保留判分器的原始结论，个别参考答案与模型表述的差异继续通过案例检查。

元数据组得到 **55/179（30.73%）**，比基础版多答对 12 题、丢失 9 题，净增 3 题（配对精确检验 p=0.664）。相对消息级 E5 Dense 和 hybrid-meta 分别少答对 3 题和 10 题。元数据组 179 题的 358 个提示哈希均重放一致，3 个回答达到长度上限。两种 Dense X 的 94 道同提示题，回答和判分全部一致。

元数据重排改变了 85 道题的前 10 条集合。已有 Part 2 标注中，18 道带明确目标消息的诊断题有 7 道能在 Dense X 的 50 条候选内找到目标；基础和元数据组的前 10 条各找到 5 道。原生及同预算原文的 179 份提示均经 tokenizer 检查，证据上限为 3,500 tokens，实际证据用量中位数分别是 2,523 和 3,431。等预算分数待另外两组完成。原始核验见金融服务器本轮目录中的 `densex-first-condition-audit.json`、`densex-meta-dev-audit.json`、`densex-dev-retrieval-audit.json` 和 `densex-native-render-audit.json`。

## RAPTOR Healthcare 构建与检索核验

课程服务器用 1,019 秒从 30,000 条消息构建了 30,007 个叶子和 1,124 个摘要节点。19 次生成达到长度上限，均在一次重试内正常返回。存盘的树、摘要记录和五组排名已核对源语料、模型和开发集身份。金融服务器用正式评分入口成功载入 33 道题的排名，原生节点和同预算原文的证据用量中位数分别为 3,367 和 3,431 tokens，均在 3,500 tokens 上限内。

摘要参与排序后，有 28/33 题的原文前十条集合与只检索叶子的条件不同。基础、元数据和叶子条件在五道带明确目标消息的既有诊断题里均未召回目标。该样本用于解释检索差异，问答效果待评分后评估。核验记录位于金融服务器本轮目录的 `raptor-healthcare-ranking-audit.json`。

## 固定来源

| 方法 | 论文与官方实现 | 本轮核查 commit |
| --- | --- | --- |
| Dense X | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.845/)、[FactoidWiki](https://github.com/chentong0/factoid-wiki/tree/b066e8b435d9de2def0983179216ae7a480b02f2) | `b066e8b435d9de2def0983179216ae7a480b02f2` |
| LightMem | [论文 v4](https://arxiv.org/html/2510.18866v4)、[LightMem](https://github.com/zjunlp/LightMem/tree/80211869f55e9c2276a7e7f564356d6cb03b7d8c) | `80211869f55e9c2276a7e7f564356d6cb03b7d8c` |
| RAPTOR | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)、[RAPTOR](https://github.com/parthsarthi03/raptor/tree/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767) | `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767` |
| HippoRAG 2 | [ICML 2025](https://proceedings.mlr.press/v267/gutierrez25a.html)、[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26) | `d5c8329422e0a0b834a15874545cb6a74b4f9b26` |

课程对照沿用 Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`、E5-small-v2 `ffb93f3bd4047442299a41ebb6fa998a38507c52`（384 维）。原版命题模型为 [propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large/tree/aaa3ddeef7ac710cdbfa3ae306fa2c6fec21884a)，约 3.13 GB。LightMem 压缩器为 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank@5f0c82792b7ea14c6484e015b6a072009496b7f2`，约 709 MB。两者已下载到本轮独立目录，权重与公开仓库的 LFS SHA256 及传输后的本地哈希一致。

更换论文中的大模型/embedding 后，称为官方管线的 GroupMemBench 本地适配，不声称复现论文的绝对分数。

## 完整工作顺序

1. **输入与接口。** 固定来源、环境和数据；上游仅接触公开消息/问题。提取器只读源消息，不能读任何题目、答案或支持消息。先用假后端检查源 ID、元数据、排序、失败和恢复。
2. **真实小测与成本剖析。** Dense X 先比较两个提取器在固定每域 50 条、共 200 条源消息上的忠实性、原子性、自足性和覆盖率。另以预先声明的代词、条件、否定等病例检查边界，不用标签选例。其他方法从连续历史窗口逐步增加到约 1,000 条，必须触发其全部核心步骤。缩小语料的小测不作为正式 QA 准确率。
3. **完整建库。** 按 Dense X → LightMem → RAPTOR → HippoRAG 2 推进，按域缓存，可中断恢复。两种元数据条件共用索引，所有重试和加载时间记账。先根据小测外推，再允许任务进入已授权预算。
4. **统一 dev。** 按下表完成配对和消融，只在固定 179 题选择。四种方法一起分析，不每做完一项就另开一轮无限探索。
5. **冻结后续对照。** 固定一个胜出方法、其无元数据对照及配置，再评估原 536 题。该集合已经看过历史结果，只能称后续对照，不能称首次盲测；如需新的确认性证据，应另选未用于开发的数据，不能重打乱旧题冒充新 holdout。
6. **收尾。** 复算配对增损和成本、检查正负案例及来源、更新报告；父 agent 审核 diff 和真实运行记录后提交。没有真实模型小测，不把新推理代码交付为已验证的全量入口。

工程错误先修复再判断方法效果。完整 dev 上输出有效但收益差时记录负结果并结束该方向，不持续堆叠方法。

## 比较矩阵

主表固定相同问题、域内语料、Qwen/E5、最终 K=10 条原消息、原 QA prompt 和 judge；保留各方法的实际索引节点、候选量、输入 tokens 和索引成本。

| 方法 | 主对照 | 元数据配对 | 单项机制消融 | 原生上下文检查 |
| --- | --- | --- | --- | --- |
| Dense X | 真正原子命题 → E5 → 父消息 max 分数 → 原文 | 同一索引 + 有界元数据重排 | 句子切分替代命题提取 | 直接使用命题 |
| LightMem | 压缩/分段/STM/LTM/sleep → 原文 | 同一记忆 + 有界元数据重排 | 去掉 sleep，其他不变 | 使用 LTM 事实/摘要 |
| RAPTOR | UMAP/GMM → 递归摘要 → 多层检索 → 原文 | 同一树 + 有界元数据重排 | 仅检索相同切分的叶子 | 叶子和摘要混合上下文 |
| HippoRAG 2 | 完整 OpenIE/filter/PPR → 原文 | 同一图和 filter 结果 + 有界元数据重排 | 关闭图传播，匹配 passage dense 控制 | 已是 passage，不重复等价条件 |

- **八组主比较**：四种方法 × 基础/元数据，每组完整 179 题。既有 BM25、Dense、hybrid、hybrid-meta 在协议、输入和运行时核对一致后作为归档控制，旧成本不归零。
- **四项机制消融、三项原生上下文检查**也使用同一 dev。完全等价的控制只在逐题提示和执行身份核对一致后复用；缓存索引一致本身不证明答案可复用。
- 元数据重排先固定简单方案：每方法产生 50 条唯一原消息候选，仅在池内根据公开的群组/阶段/主题和明确目标人物排序。复用现有提示解析，不把 asking user 一律当目标人物。无有效提示时基础/元数据排名完全相同；确切规则在模型 dev 判分前固定，不做大规模权重网格搜索。
- 原生上下文单独成表：命题/摘要/树节点与十条原文不是同一单位。这些条件及配套原文控制统一采用同一 tokenizer 的 3,500-token 检索上下文上限，记录实际数量与截断。QA/judge 模型、问题和生成规则保持一致，输入格式差异披露。
- 默认仅一个胜出方法及其对照进入 536 题后续评估。任何看过 test 后的新改动必须重新标为探索。

此表是完整工作清单，不是预算内必能完成的承诺。主要未知成本在建库，不能按 179 次问答估算。若外推超出授权或课程上限，先协商资源/范围，不静默缩语料再写“全量”。

## 各方法实现约束

### Dense X

官方仓库发布提取模型和例子，没有完整可移植流水线。句子切分只作消融。FLAN-T5 与 Qwen 在同一源消息样本上比较质量/成本后选一个，不用 QA test 选提取器。

初始配置仅输入当前正文：T5 的 Title/Section 留空；Qwen 保留条件、否定、不确定性，缺少上下文时不猜代词主体。这对对话自足性的影响必须实测，加入说话人或上下文须另记为适配。权威元数据从原消息回溯。

保留完整输出和状态；JSON 错误、空数组、截断显式标记并保留原文回退项，运行时故障中止并支持恢复。单位 E5 打分后先按父消息取 max，再取 50 个唯一父消息；不能先截取 50 个命题再去重。分别报告命题命中和原文回退率。

### LightMem

使用 [LightMemory](https://github.com/zjunlp/LightMem/blob/80211869f55e9c2276a7e7f564356d6cb03b7d8c/src/lightmem/memory/lightmem.py) 的 LLMLingua-2 → 注意力/相似度联合分段 → STM → flat 抽取 → LTM → 离线 sleep。不把批量摘要检索改名为 LightMem，也不引入 StructMem。

按官方 LoCoMo 方式将真人消息映射为 user + 空 assistant 的轮次对，保留真实 speaker/dia_id；按 `(domain, group)` 隔离缓冲及更新，检索时合并域内全部可见群组。来源编号不能越界后裁剪成伪引用。sleep 保存更新前后及参与的源 ID。

父 agent 已核对源码中的末尾分段覆盖、可能跳过 STM flush、线程结果未消费和改写事实仍用旧向量。先用夹具复现并做最小修正。重新嵌入改写事实会影响检索，需明确披露；旧行为仅小样本对拍，不为此重复全量。Qdrant 使用独立持久化路径，避免默认初始化清除旧目录。

### RAPTOR

使用 [ClusterTreeBuilder](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py) 的 UMAP/GMM 软聚类、BIC 簇数和递归摘要。叶子在每条消息内部切分并保留原 ID；父节点保留完整后代来源，不能拼接整域文本后丢消息归属。

注入本地摘要模型、分别处理 query/passage 前缀的 E5 编码器。检查小簇、无缩减递归、过长簇、摘要截断和确定性。上游 UMAP 默认没有 random_state，模块级 random.seed 不足以证明重复构建一致。原生摘要上下文和回溯原文分开报告。

### HippoRAG 2

使用 [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/HippoRAG.py) 的实体/三元组、passage/phrase 图、关系/包含/同义边、query-to-triple、recognition filter 和 PPR。online 路径连接本机 Qwen 即可，完整保留 few-shot prompts，不用单轮简化接口丢掉示例。

实测 120,000 条原消息有 119,620 个域内唯一正文。上游按正文 hash 去重并覆盖 metadata，适配 sidecar 必须保留全部原 ID 和来源，不能只取最后一个 speaker。filter 合法空集合可以走官方 dense 回落，解析/请求错误必须显式失败，不能伪装正常图检索。直接读结果列表，避免 `to_dict()` 仅保留五项。

NER 和 triples 合计约 **239,240 次逻辑调用**，尚未计重试/filter。batching 能加速但不能忽略成本。先测吞吐、同义图边密度、RAM/显存再外推；E5 分数分布不同，默认阈值 0.8 不能直接当作适用。最多一个预先声明的小范围 dev 阈值检查，不扩展为无界图参数搜索。

100 条源消息的小测后，预先声明该检查仅为 **0.8 与 0.9** 两个阈值，共用相同实体/三元组和 E5 向量；其他图与问答参数不变。先完成成本估算，再按同一 179 题 dev 比较来源检索与问答，并冻结一个阈值。上面的 0.85/0.95 只读边数诊断不进入 QA 参数搜索。保留 `Msg_845` 疑问变事实、`Msg_291` 否定反转，以及日期/SLA 错连作为质量核验病例，不按这些自查询的返回名次选型。

在 1,000 条试跑消息的已保存图上，按实际边来源只读移除低于 0.9 的纯同义边，边数从 335,847 降到 37,881，24 个连通分量保持一致。316 条同时拥有事实或段落贡献的边仍保留其原权重。这个检查只测图的稀疏度，正式选阈值仍需固定开发题的检索与问答结果。

## 运行和证据

上游各自独立环境产生索引和带来源的检索结果，主表继续使用现有 in-process Transformers QA/judge。RAPTOR 固定的 Transformers 4.38.1、HippoRAG 的 4.45.2 不能直接拿来推理 Qwen3，不覆盖 `requirements-local.txt`。服务限定本机地址，核验 JSON、非思考模式、SDK 和 usage，不调用付费外部模型。

正式记录上游/适配代码 commit、模型/data revision、配置、索引身份、实际依赖、硬件、分阶段时间、逐题轨迹。恢复必须核对身份；错误、不完整与合法回退分开。GPU-hours 包括各卡加载、建库、压缩、空转、重试和 QA；CPU 时间、RAM、磁盘另列。

输出整体/分域/六类准确率、配对增损、来源覆盖、非法引用、解析/回退率、截断、索引/在线时延和资源。支持消息仅作事后诊断。先排除实现、prompt、缓存和判分问题，再解释模型或方法局限。

## 已有离线入口

```bash
# 仓库根目录；只导出公共数据，不加载模型。
python scripts/prepare_memory_inputs.py --output results/memory-inputs-dev
```

输出 `messages-<Domain>.jsonl`、`queries-dev.jsonl`、200 条源消息的 `extraction-audit.jsonl` 和清单。查询不含答案/支持证据，ID 和 category 仅用于定位，检索输入仍限 question/asking_user。相同正文保持不同 `(domain,id)`。默认不导出 test，拒绝覆盖输出目录。

已离线核验四域各 30,000 条消息，群组数 6/7/10/10，同文重复出现次数 100/129/68/83；完整公共输入导出、179 题覆盖及答案字段排除通过，模型调用 0。临时全量导出已清理。

`scripts/build_propositions.py` 支持本地 FLAN-T5/Qwen、真实批处理、显式原文回退及断点恢复。批大小、模型配置与权重、输入、代码和实际运行时纳入恢复身份；同一批输出按源消息顺序保存。首个 EOS 后的 padding 不作为截断，超长输入不静默裁剪。整体成本使用包含模型加载的会话时间，逐条 `seconds` 只是批耗时的分摊。强制中断时无法恢复的局部计时不按零处理，外层资源账本仍计全部占卡时间。

命题小测均使用固定每域 50 条的同一批 200 条样本，没有读取题目标签。下表是 builder 会话时间；整卡预算另由外层账本记录，包含服务启动及共享该服务的其他任务，不能把两层时间相加：

| 小测 | 解析有效 / 原文回退 | builder 会话时间 | 观察 |
| --- | --- | --- | --- |
| 初始 Qwen、单条生成 | 181 / 19 | 468.31 秒 | 12 条返回对象数组、7 条空数组；发现提问被改写为事实 |
| 官方 FLAN-T5、batch=8 | 199 / 1 | 112.17 秒 | 1 条输出截断；发现把征求共识改写为已经达成共识 |
| 保留语气的 Qwen、Transformers batch=4 | 200 / 0 | 421.15 秒 | 消除了该样本的格式回退，仍有条件和受话人误读 |
| 同一提示、Qwen HTTP batch=8 | 200 / 0 | 191.06 秒 | 连续批处理加速；运行时不同，不声称输出逐字一致 |
| 补充条件/请求约束、Qwen HTTP batch=8 | 200 / 0 | 191.67 秒 | 修复部分遗漏和角色误读；仍有条件丢失和歧义误解 |

以上是解析和源内容质量诊断，不是问答准确率。Qwen 在已检查的共识、纠正和条件案例中总体更适合对话，仍会把 Finance/Msg_13407 的条件写成事实，误读 Manufacturing/Msg_8821 的省略表达；FLAN 也有共识幻觉和因果反转。这是有针对性的源内容审计，不能由此估计总体语义错误率。历史小测代码与实际清单核对后保留必要证据，不能用改进代码冒充初始运行。

连续历史小测固定为 Finance 的 AML 群组按时间和 ID 排序的前 100 条消息，仅用原消息正文自查询检查接口，输入 SHA256 为 `bcc1fc2b97b3db9a33c20f8e2094d841101261fdd336e81569e0bd1b5f1644d5`。

| 最终小测 | 实际触发的机制 | 会话时间与结果 |
| --- | --- | --- |
| LightMem | 压缩 100 条、14 次分段、54 次轮次编码、13 次抽取、59 次 sleep | 663.43 秒；115 条 sleep 前事实、108 条整合后事实；41 改、7 删、11 忽略 |
| RAPTOR，8 个摘要 worker | UMAP/GMM 软聚类、两层递归摘要 | 45.73 秒；`137 → 29 → 5` 节点，34 次摘要调用 |
| HippoRAG 2 | 100 次 NER、100 次 triples、建图、3 次筛选和 PPR | 299.59 秒；762 phrase + 100 passage，952 条唯一规范化 triples |

这些会话并非全部在独占卡上运行，时间用于阶段诊断，不能直接当整卡吞吐。RAPTOR 串行前测为 213.07 秒、`137 → 29 → 4`、33 次摘要；并发调用下实际模型输出和上层聚类可以变化。固定假摘要的测试证明节点顺序不受线程完成次序影响，未声称不同服务调度下真实生成逐字一致。

LightMem 已修复两个实际问题：使用源消息的真实时刻建立严格较新的更新关系；长事实无损分片编码后按正文 token 数加权聚合。复测核对 100/100 条进入抽取、116 条候选更新边均符合原始时间顺序。607-token 的 `Msg_88` 分成 512/100-token 两片，9 个来源全部保留，独立 CPU 重算的聚合向量与存盘逐元素一致。sleep 前来源覆盖 100/100，之后为 97/100；模型仍会误删补充信息、遗漏否定或误改因果。前测 95/100 与本次 97/100 的差异不是语义准确率，也不能归因于代码修正。

HippoRAG 已阻止四字段关系、无限重复和 few-shot 示例实体混入图。完整原始响应与丢弃诊断保留；仍有把疑问写成事实等内容错误。E5 默认 0.8 同义阈值产生 43,617 条边，占图边的 95.76%；0.85/0.90/0.95 的只读密度探针为 6,088/986/110。高阈值仍会把不同日期或时长看作相似。Healthcare 开发集的 0.90 对照复用 0.80 的图和识别缓存，单独核对稀疏化是否改变检索。[运行方式](hipporag-integration.md)

四种索引通过 `scripts/retrieve_memories.py` 导出基础/元数据配对；每题共用一次检索所得的 50 个原消息候选，固定 RRF 60，最后 K=10。`scripts/run_local_baselines.py --precomputed <排名目录>` 接回原 QA/judge，校验完整领域源消息、题目原文、提问者、题号和排序文件；回答模型只读取 ID 对应的原消息。命题、摘要和图的诊断轨迹不进入主表 QA。离线回归已证明同一有序原消息生成逐字相同的回答/评分调用；这不替代真实模型测试。

例如，使用同一份 Healthcare 排名对照 RAPTOR 和 RAPTOR + 元数据。`QWEN_MODEL`、`E5_MODEL` 和 `RAPTOR_RANKINGS` 分别指向本地冻结权重和包含 `raw-top10-pair/` 的 Healthcare 排名目录。

```bash
python scripts/run_local_baselines.py \
  --model "$QWEN_MODEL" --model-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --embedding-model "$E5_MODEL" --embedding-revision ffb93f3bd4047442299a41ebb6fa998a38507c52 \
  --methods raptor raptor-meta --precomputed "$RAPTOR_RANKINGS/raw-top10-pair" \
  --domains Healthcare --full --partition dev --output results/raptor-healthcare-dev
```

做接口小测时，将 `--full` 换成 `--categories temporal --limit 1`，并使用新的输出目录。评估四域时给出四个领域和各自的排名目录。每个领域、每种条件只传入一份排名。

同一导出入口接受四个原消息消融名称。它们各生成一份独立排序，均保留 50 条唯一原消息候选，最终取 10 条交给原 QA。

| `--method` | 检索条件 | 索引输入 |
| --- | --- | --- |
| `densex-sentences` | 确定性句末切分，E5 逐句打分后取父消息最大分 | 原消息和相同 E5 快照，省略 `--index` |
| `lightmem-pre` | sleep 前快照 | 与 `lightmem` 相同的索引目录 |
| `raptor-leaf` | 仅以同一树的叶子参与打分 | 与 `raptor` 相同的树和模型文件 |
| `hipporag2-dpr` | 官方 passage dense 排序，共用原消息映射 | 与 `hipporag2` 相同的已完成索引目录 |

句子对照只按正文的句末标点切分，保护常见缩写、人名首字母及数字内部的小数点，保留引号和原消息溯源。它与命题条件使用同一父消息聚合，无生成模型提取调用。HippoRAG 的 DPR 对照不调用 recognition filter 或 PPR。同一适配器版本下两个条件共用一份索引，现有源码哈希和语料哈希校验仍适用。

`densex`、`lightmem` 和 `raptor` 可用 `--context-view native` 或 `--context-view raw-budget` 导出各自的上下文对照。每次导出生成一份 `<method>-<view>.jsonl`，每题至多 50 个带原消息 ID 的候选。`native` 按实际命题、记忆条目或树节点排序，多个原生单元可以来自同一条原消息。`raw-budget` 使用相同方法排序后的唯一原消息。导出阶段保留完整候选和来源，问答回放渲染提示后才应用 3,500-token 预算。

运行目录为金融服务器 `/home/ai4s_group/XZG/NLP/sgmem-memory`；模型、上游 checkout、增量依赖和运行产物位于 `/data/ai4s_group/XZG/NLP/memory-round-20261124`。旧 `sgmem-part3` 未修改。复用原 Python 3.10 环境，缺少的依赖以独立 `deps/` 目录加入 worker 的 PYTHONPATH，不覆盖已有 Torch/Transformers。直接依赖见 `requirements-memory.txt`，实际版本另存运行目录。服务器 `/home` 空间紧张，本轮缓存和临时文件均放 `/data`。

`scripts/run_memory_gpu.py` 双重检查候选卡，优先整张空闲卡，其次使用达到该阶段显存门槛的共享卡。单个账本用互斥锁和 GNU timeout 约束阶段用量。两台服务器的已用时长及未结束阶段预留需合并计入本轮 30 GPU-hours，预留以金融服务器本轮目录的 `second-wave-budget.json` 为准。`scripts/run_memory_service.py` 将本机 Qwen 服务绑定到 worker，固定非思考模板并验证与 tokenizer 一致，结束时释放服务进程组。服务启动、等待工作及失败计入占卡时间。

真实小测入口为 `scripts/smoke_memory_pipelines.py`，使用预先固定的连续公开历史窗口。验收区分“能建库”和“机制实际触发”：RAPTOR 必须生成摘要层；LightMem 必须执行压缩、分段、轮次相似度、事实抽取和 sleep 调用，另记 update/delete 数量；HippoRAG 必须至少有一次经过筛选与 PPR 的检索。小语料自查询仅检查接口和来源，不能当作 dev 准确率。

父 agent 复核了三位子 agent 的代码与固定上游 API，并修复/要求修复了源 ID 丢失、错误吞掉、改写文本未重新嵌入、图筛选任意模糊匹配、模型身份校验和服务清理问题。实际模型质量、全部核心流程、成本与最终分数分别验收；CPU 测试通过不等于方法已成功。
