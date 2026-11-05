# 检索实现参考

以下是实现与受控实验的依据，不代表已经实现或验证了性能提升。基线、方法变体和所有对照遵循[实验协议](experiment-protocol.md)。

## 词法检索：rank-bm25

[rank-bm25](https://github.com/dorianbrown/rank_bm25) 提供 BM25 实现，适合通过 Python 包依赖使用。固定的 GroupMemBench 子模块基线使用 `BM25Okapi`，按每条消息的 content 建索引，使用小写和 `[A-Za-z0-9_]+` 分词，不去除停用词；先核对并保持这些细节，才能有可解释的本地基线。基线本身的模型与 judge 与本项目本地方案不同，不应称作官方配置的精确复现。

后续可在开发集单独比较分词、BM25 参数或候选池大小，记录 ID、长度、空查询及并列排序的行为。改变基线设置是实验变量，应分别报告并保留原基线，不预设这些变化会改善结果。

## 稠密检索：E5 与 Sentence Transformers

[E5 官方代码](https://github.com/microsoft/unilm/tree/master/e5) 提供模型和检索评测参考；[Sentence Transformers](https://github.com/huggingface/sentence-transformers) 提供文本编码及检索相关工具。参考模型 [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) 的官方用法：非对称检索分别对问题和文档使用 `query: ` 与 `passage: ` 前缀；统一记录文本截断、池化、归一化、相似度、批量大小和模型/tokenizer revision。该模型面向英文，输入最长 512 tokens；不要假设所有 E5 变体具有相同的输入要求。

消息向量可在相同语料与编码配置下复用；查询仍需单独编码。检查索引和缓存是否与数据、模型与预处理匹配，再比较吞吐、检索延迟和成本。CPU/GPU、精度和搜索实现变化也应记录，不能默认结果完全一致。

## 方法改进与对照

先建立可复现的 BM25 和 E5 Dense，分析 speaker/group/reply/time 相关失败，再评估元数据重排序。各方法保持相同问题集、可见历史、最终 K、上下文预算、QA 和评测；根据需要分别测试 BM25+SGAR 和 Dense+SGAR。内容分数与元数据分数的范围、归一化与权重在开发集确定，冻结后再评测 test；按特征消融并报告成本与新错误。关键词分词、embedding 与重排序各自带来的变化应分开验证。

这三个仓库是方法参考和候选包依赖；除非需要修改或直接运行其源码，否则无需作为 Git 子模块。E5 模型权重按模型来源和许可单独管理。
