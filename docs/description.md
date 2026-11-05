Project description
Most existing long-term memory systems for LLM agents are designed for one-to-one
conversations. However, real-world interactions often involve multiple participants
communicating across group chats, meetings, or collaborative channels. In such
settings, remembering conversation content alone is insufficient. A memory system
must also preserve who said what, in which group, and under what conversational
context.
This project investigates long-term memory mechanisms for multi-party conversations
using GroupMemBench and EverMemBench. Students will study whether explicitly
incorporating speaker identity, group information, reply structure, and temporal
information into memory retrieval can improve downstream question answering. In
particular, the project focuses on failures caused by treating multi-party conversations
as independent text chunks, where information from different speakers or groups can
easily be confused.
The central research question is whether speaker- and group-aware memory
representations and retrieval mechanisms can improve long-term memory
performance in multi-party conversations compared with conventional content-based
retrieval systems.
Tasks
Part 1: Reproduce basic memory retrieval in multi-party conversations
(20%)
Use GroupMemBench as the primary benchmark and reproduce the two retrieval
baselines provided in its official repository:
BM25 Retrieval: retrieves historical messages based on lexical similarity.
Dense Embedding Retrieval: retrieves historical messages using embedding model
and vector similarity.
In the released baselines, retrieval is primarily performed over message content,
while speaker and role metadata are attached to retrieved messages before they are
provided to the QA agent.
Use the official evaluation procedure to report overall accuracy and category-level
accuracy across the six question types: multi-hop reasoning, knowledge update,
temporal reasoning, user-implicit reasoning, term ambiguity, and abstention.
Students are recommended to begin with the Finance and Technology domains,
which provide solvability-filtered question sets, before optionally expanding to other
domains.
Part 2: Analyze speaker- and group-related memory failures (40%)
Analyze failed examples from the baseline systems and investigate whether the
failures are associated with insufficient use of multi-party conversational structure.
The analysis may include:
Speaker confusion: information from one participant is incorrectly attributed to
another.
Group or channel confusion: information from one conversation group is incorrectly
applied to another group.
Incorrect belief attribution: a statement made by one participant is treated as
another participant's belief or preference.
Loss of reply context: a message is retrieved without correctly identifying which
previous message it responds to.
Temporal update errors: outdated information is used after a participant has
updated or corrected it.
Cross-participant reasoning failures: information from multiple participants is not
correctly connected during reasoning.
Students should construct a failure taxonomy, report the distribution of different failure
types, and provide representative examples. They should also investigate whether
speaker, group, timestamp, and reply information could help recover the correct
supporting memories.
Part 3: Speaker- and group-aware memory modeling (30%)
Based on the failure analysis, design and implement one focused modification to the
baseline memory representation or retrieval mechanism.
A possible structured memory record contains five fields:
Memory = (speaker, group, time, reply-to, content)
Possible directions include:
speaker-aware retrieval or reranking;
group-aware memory retrieval;
metadata-aware indexing or filtering;
structured retrieval using speaker and group relations;
a lightweight conversational graph connecting speakers, messages, replies, and
groups.
The proposed method should use conversational metadata during memory selection
or ranking, rather than merely attaching metadata after content-based retrieval.
Compare the proposed method with the original BM25 and dense retrieval baselines
under the same backbone LLM, embedding model, retrieval budget, and evaluation
setting.
Report overall accuracy and category-level accuracy. Students should also analyze
whether the proposed method retrieves more appropriate memories and reduces
speaker- or group-related errors.
Use separate development and evaluation examples, and do not use reference
answers or evaluation labels during retrieval.
The final submission should include runnable code, experimental results,
representative failure cases, and a report of approximately 6 pages covering the
methodology, experiments, and conclusions. A carefully analyzed negative result is
acceptable. Recommended sections include an abstract, related work, methods,
experimental results, and conclusions.
Part 4: Final report and demonstration — 10%
Submit a report of approximately 6 pages covering motivation, reproduction, error
analysis, the proposed repair method, experiments, failure cases, and conclusions,
together with runnable code and representative trajectories. Recommended sections
include an abstract, related work, methods, experimental results, and conclusions.

---

# 本仓库的实施说明

上文保留课程项目描述。以下说明本仓库如何对应上述要求；若实现或实验与课程原文存在差异，以原始课程要求为准。

## 项目范围与研究问题

主基准为 GroupMemBench，优先使用 Finance 和 Technology；EverMemBench 用于后续扩展。研究重点是：让 speaker、group、时间和回复关系参与记忆选择，是否能改善多人对话问答，并减少说话人或群组归属错误。

| 阶段 | 本仓库的工作 | 所需证据 |
| --- | --- | --- |
| Part 1 | 对同一可见历史执行 BM25 和 E5 Dense 内容检索，再使用统一 QA 和 judge | 总体、各领域和六类问题的准确率及逐题轨迹 |
| Part 2 | 人工分析基线失败，按 F1–F8 分类并记录支持证据 | 各类型数量、分母、代表案例和可能有帮助的元数据 |
| Part 3 | 依据开发集失败分析设计一个聚焦的元数据重排序或结构检索方法 | 在相同评测条件下比较基线、所提方法和必要消融；同时展示改善和新错误 |
| Part 4 | 汇总方法、实验、错误及局限 | 可运行代码、实验结果、代表性轨迹和约六页报告 |

当前已完成四域两种基线的 715 题全量实验，并提供固定开发样本的结构性失败分析，见 [Part 1/2 全量报告](part12-full-results.md)。两法使用相同模型、top-10 和 512/256 输出预算；模型上下文上限一致，实际 QA 输入均在上限内，没有静默截断。不同检索内容的输入长度不强行补齐。Part 3 修复方法及最终课程报告仍需完成；金融服务器结果采用直接 Transformers 后端，不把同学原环境的 Docker 验证当成这次服务器的验证。

## 公平比较与防泄漏

- 使用相同的问题集合、允许检索的历史、最终 Top-K、上下文预算、QA 模型和评测规则。不同模型、索引或运行预算产生的结果需明确说明，不能直接归因于检索算法。
- 参考答案和人工证据只用于判分与事后分析，不能进入检索请求或参与特征构建；只在开发集选择方法和参数，在评估集上报告冻结后的结果。
- 默认划分以领域和问题类型分层，按问题 ID 区分开发集与测试集；两者共享历史对话，不能证明跨对话泛化。应核查课程或官方基准是否要求其他划分方式。
- 问答与 judge 判分均通过可配置的本地 OpenAI 兼容服务调用。使用本地模型及 E5 时，实验属于对官方流程的本地适配；上游原始配置的模型和 embedding 不同，不能声称分数为精确复现。

## 数据流与失败分析

`GroupMemBench 消息 / 问题 → 内容检索 → 格式化包含元数据的上下文 → QA → judge → 逐题结果与指标 → 人工失败标注`

原始 BM25 和 Dense 检索只根据消息内容排序，元数据在检索后用于 QA 上下文。后续改进应在最终选择或排序时使用允许访问的元数据，而不是只改变上下文显示。询问者与目标人物不一定相同；未知群组不能由参考答案推断，较新消息也不一定覆盖旧消息。

失败分析应区分检索阶段未取得足够证据和证据已取得但 QA 推理失败。人工标注可记录多个错误类型，但须指定一个主类型和标注者；报告标注覆盖率及多标签计数的分母。judge 不明确或服务调用失败应单独列出，不能自动推断错误类型。

## 本机与 HPC 运行

[README](../README.md) 给出子模块初始化、Python 依赖安装、划分、基线运行及人工标注命令。运行前需要可访问的 Qwen 等本地模型服务，以及 Dense 所用的 E5 模型文件；两条基线应使用同一 QA/judge 配置。运行过程会生成参考答案和检索记录，分享前应核查数据许可和访问范围。

[Dockerfile](../Dockerfile) 用于构建项目 Python 环境；本机镜像已构建，容器内 PyTorch 已检测到 RTX 3090；Qwen 与两条基线已在容器内通过同一道开发集题目的端到端验证，尚无正式实验结果。模型、基准数据和实验结果应通过挂载或独立服务提供。HPC 上使用 Docker 或 Apptainer 取决于实际集群配置，GPU 驱动、容器运行时、网络和模型缓存必须在目标环境验证，详见 [HPC 部署说明](hpc-deployment.md)。临时测试输出按 [AGENTS.md](../AGENTS.md) 在对话结束前清理；正式报告依赖的结果与轨迹按实验协议保留。
