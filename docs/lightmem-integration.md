# LightMem 在 GroupMemBench 上的适配

适配入口使用官方 LightMem 提交 `80211869f55e9c2276a7e7f564356d6cb03b7d8c` 的 `LightMemory`。建库依次运行 LLMLingua-2、注意力和相邻轮次相似度分段、512-token STM、flat 事实抽取、LTM 直接入库、睡眠队列与离线更新。使用 LoCoMo 的压缩设置 `rate=0.6` 和默认睡眠相似度阈值 `0.9`。参照 [LightMem 论文 §3](https://arxiv.org/html/2510.18866v4#S3) 与 [官方源码](https://github.com/zjunlp/LightMem/tree/80211869f55e9c2276a7e7f564356d6cb03b7d8c)。

Healthcare 扩容时，部分 sleep 请求的输入超过 4,096 tokens，加上预留输出后被 8,192-token 服务拒绝。恢复运行使用 `scripts/run_memory_service.py --context-length 16384`，实际设置写入服务清单。每次仍先请求最多 4,096 个输出 tokens；仅在 `finish_reason=length` 时保持原提示和种子重试一次，上限 8,192，并累计两次 token 用量。缺少 usage、其他结束原因或第二次仍截断都会明确失败，错误包含结束原因、用量和响应尾部。索引清单记录该生成策略；新策略使用独立输出，不能与旧群组快照混为同一协议。

启用重试后，指标 `llm.*.calls` 表示成功完成的逻辑请求数，不能直接标成实际 API 请求次数。token 用量包含两次尝试；实际 HTTP 请求次数应另查该阶段服务日志，GPU 账本仍包含加载、重试和失败耗时。

每个群组的原消息按实际时刻和 ID 送入独立缓冲区及 LTM，真实说话者沿用 LoCoMo 的 `user` 加空 `assistant` 表示。抽取与睡眠使用源消息时刻，带时区输入转换为 UTC，无时区输入按 UTC 解释。睡眠队列在同一群组内先过滤严格早于当前事实的时刻，再按相似度取近邻。同一时刻的事实之间没有更新关系。检索时用统一 E5 相似度比较该领域的所有群组。库内事实的抽取锚点必须属于传给模型的批次。改写后的事实保留原锚点和睡眠候选的源 ID。这个集合记录实际输入的出处，无法判定模型采纳了其中哪些信息。原消息文本、说话人、群组、原时间字符串、角色及回复关系由 `sources.jsonl` 提供。

## 环境

官方项目声明 Python 3.10–3.11。核心路径直接需要 NumPy、Torch、Transformers、Sentence Transformers、LLMLingua-2、Qdrant Client、Pydantic、OpenAI、HTTPX、tiktoken 与 PyYAML。上游 [依赖声明](https://github.com/zjunlp/LightMem/blob/80211869f55e9c2276a7e7f564356d6cb03b7d8c/pyproject.toml#L10-L81) 的主要固定版本如下。

| 组件 | 官方固定版本 |
| --- | --- |
| Torch / Transformers / Sentence Transformers / NumPy | 2.8.0 / 4.57.0 / 5.1.1 / 2.2.6 |
| LLMLingua / Qdrant Client / Pydantic | 0.2.2 / 1.15.1 / 2.12.0 |
| OpenAI / HTTPX / tiktoken / PyYAML | 2.3.0 / 0.28.1 / 0.12.0 / 6.0.3 |

在隔离的 LightMem worker 环境中使用已经过验证的 Torch/CUDA 配置。若复用 SGMem 的 Torch 2.8 和 Transformers 4.57.1，清单记录实际版本，端到端小测用于核实兼容性。Qwen 通过已有的本地 OpenAI 兼容端点服务。它须支持 `response_format=json_object`、`chat_template_kwargs.enable_thinking=false`、`seed`，并返回 `finish_reason` 与 token usage。启动服务时应单独记录其实际权重修订。服务名和 `/v1` 地址仅标识调用入口。

本地模型路径为课程冻结的 E5-small-v2 `ffb93f3bd4047442299a41ebb6fa998a38507c52`，以及官方默认压缩器 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` `5f0c82792b7ea14c6484e015b6a072009496b7f2`。适配器检查冻结 E5 权重哈希，记录压缩器哈希和调用方提供的压缩器修订。Qwen3-8B 应使用课程冻结修订 `b968826d9c46dd6066d109eabc6255188de91218`。

LLMLingua 初始化会读取 `cl100k_base`，STM 会读取 `o200k_base`。离线运行前，在 worker 使用的 `TIKTOKEN_CACHE_DIR` 中准备这两份 tiktoken 数据。

## 建库和检索

在固定上游提交的独立 checkout 中应用补丁。使用 `prepare_memory_inputs.py` 导出的域内 `messages-<Domain>.jsonl`，或使用它的同 schema 公开消息小样本。

    git -C "$LIGHTMEM" apply /absolute/path/to/sgmem/patches/lightmem.patch
    PYTHONPATH=src python -B -m sgmem.lightmem_memory "$MESSAGES" "$LIGHTMEM_RUN" \
      --upstream "$LIGHTMEM" --endpoint http://127.0.0.1:8000/v1 --model "$SERVED_QWEN" \
      --embedding-model "$E5_MODEL" --compressor-model "$COMPRESSOR_MODEL" \
      --compressor-revision 5f0c82792b7ea14c6484e015b6a072009496b7f2

每个群组的 Qdrant 状态驻内存。完成的群组作为包含前后快照、向量、更新记录和计量的单个 NPZ 原子保存。输出目录只接受一个建库进程。使用相同输入及参数增加 `--resume` 可重试失败的群组，已完成快照被复用。`attempts.jsonl` 记录每个正常返回或抛错的尝试及墙钟时间。进程强制终止时，应另用外部 GPU 预算日志核算。每个前后事实向量共约 3 KiB，另外有事实及来源记录和原始消息边车。

    from sgmem.lightmem_memory import LightMemIndex

    index = LightMemIndex.load(run_dir, "Finance", stage="post", device="cpu")
    hits = index.retrieve(question, 10, asking_user_id=asking_user_id)
    facts = index.retrieve_native(question, 10, asking_user_id=asking_user_id)

`retrieve` 按事实得分回溯并去重原消息，返回现有 QA 接口接受的 `message`、`score`，并附 `unit_id`。原消息候选不足 K 时显式报错。`retrieve_native` 返回带锚点及来源的事实单元，原生上下文另用同 tokenizer 预算评测。`stage="pre"` 使用睡眠前的快照供消融。

补丁修复尾部清空、线程异常传播、无效事实锚点、合成时间偏移，以及更新文本后仍沿用旧向量的问题。按每个群组生成稳定事实 ID，通过反向列表保留更新来源和候选顺序。

E5 使用 `passage:` 入库，使用 `query:` 查询和对称轮次编码。限制为 512 token，包含前缀和特殊 token。超长事实使用 tokenizer 字符偏移切成连续片段，拼接片段可恢复原文。每片独立加 `passage:` 前缀并编码，以片段正文的 token 数为权重平均归一化向量，最后再次归一化。索引仍保留一个完整事实及其来源。查询或轮次编码超过上限时显式报错。索引清单保存时间与长度策略，早于这些策略生成的向量须重新建库。该实现保留官方核心流程，报告时应说明 GroupMemBench 语料、模型、出处投影与重嵌入配置。

## 验收

本地 CPU 假后端测试覆盖完整适配调用顺序、尾部清空请求、分群、更新出处、时间并列、超长事实末尾参与向量计算、恢复、原消息去重及本地端点限制。真实小测需核查清单中的 `changed_turns`、`segmentation_calls`、`released_segments`、`embedding_calls.turn`、`llm.add_memory.calls` 和 `update_actions`。建库验证每条输入消息按顺序交给抽取器，并记录 `extraction_source_turns`、两个快照中可回溯的唯一锚点数及全部唯一来源数、`embedding_length` 的超长事实与片段数。抽取器收到消息并不保证产生事实，睡眠删除也可能降低来源覆盖。若相似度阈值下无睡眠候选，受控例子应补齐更新或删除覆盖，并保存真实资源计量。正式检索及 QA 沿用固定题目和 K10 原消息口径。
