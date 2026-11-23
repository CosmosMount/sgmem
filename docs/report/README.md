# 课程报告与离线演示

2026-11-23：基于冻结实验整理的 **6 页英文报告初稿**，供组员审核后提交。本文不代表已经提交课程作业；成员姓名、贡献和独立人工复核尚待小组补充。

- 阅读 [report.pdf](report.pdf)；编辑 [main.tex](main.tex) 和 [references.bib](references.bib)。
- 下载后用浏览器直接打开 [case-demo.html](case-demo.html)，无需服务器、联网、模型或 GPU。GitHub 网页可能显示源码，需下载后打开。
- 最终 test 为 BM25 178/536、Dense 146/536、hybrid 164/536、hybrid-meta 177/536。元数据相对 hybrid 净增 13 题，p=0.111；不能表述为显著超过 hybrid 或超过 BM25。

## 文件如何生成

在仓库根目录执行。绘图只需要既有 `analysis` 依赖（Matplotlib）；HTML 脚本只使用 Python 标准库。均读取已保存的实验结果，不加载模型或重新判分。

```bash
# 如需在新的分析环境安装绘图库：python -m pip install -e '.[analysis]'
python scripts/build_report_assets.py
python scripts/replay_cases.py
python scripts/replay_cases.py --list
# 只生成一条案例到临时目录：
python scripts/replay_cases.py --case Technology/multi_hop_24 --output /tmp/sgmem-case.html

# 需已安装 XeLaTeX、latexmk、BibTeX 和文中字体：
cd docs/report
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
# 审查 build/main.pdf 后再更新交付文件：
cp build/main.pdf report.pdf
```

Windows 已在 Conda `latex` 环境编译验证：从仓库根目录使用 `conda run -n latex python scripts/build_report_assets.py`，进入 `docs/report` 后使用 `conda run -n latex latexmk -xelatex -outdir=build main.tex`。本次绘图为 Matplotlib 3.11.2；编译字体为 TeX Gyre Termes、TeX Gyre Heros、Latin Modern Mono 和 TeX Gyre Termes Math。`build/` 中的中间文件不提交；三张矢量 PDF、生成的 `results.tex` 和审查后的 `report.pdf` 随源码交付。

本次验收：CPU 测试 **121 项通过**（包含两项回放检查）；三张图和结果表在临时目录重建后逐字节一致；PDF 共 6 页，无未解析引用、缺字或越界警告，并逐页检查排版；浏览器检查了检索列表、judge 输出和源消息的展开。案例回放标签始终表示原始模型判分。

| 展示内容 | 唯一实验依据 |
|---|---|
| 八组开发集比较与成本散点图 | [`structured-qa-dev.json`](../evidence/structured-qa-dev.json) |
| 536 题总体、六类准确率与逐题增损 | [`structured-test-comparison.json`](../evidence/structured-test-comparison.json) 与 [Part 1 metrics](../evidence/part12-full-20261105/metrics.json) |
| 38 条失败分类图 | [annotations](../evidence/part12-full-20261105/failure-annotations.jsonl) 与 [failure-analysis](../evidence/part12-full-20261105/failure-analysis.json) |
| 历史成本 | 基线 provenance、两轮 reply-dev 记录、metadata 两份外层 budget ledger，以及新补齐的 [历史计时凭证](../evidence/historical-resource-receipts.json) |
| 六条演示的回答、前十 ID、judge 输出和源消息 | [`structured-test-cases.json`](../evidence/structured-test-cases.json) |

记录完整时长的非重叠运行小计为 **13.3489 GPU-hours**，包含新增找到的四条 pilot/预检计时记录。缺少原始时长的历史测试仍未计入；这不是整个项目的精确总成本。八组 dev 和两组 test 已包含在 metadata 外层账目中，不重复相加。

## 三分钟演示与队友复核

1. 先说明页面是历史结果回放，再打开 Technology/multi_hop_24：展开 `Msg_27529`，展示支持证据进入第 5 位后，回答从弃答变正确。
2. 展示 Technology/term_ambiguity_30：同一个方法也会挤掉原第 10 位的正确证据。回答增益应同时看 35 gain 和 22 loss。
3. 展示 Healthcare/temporal_23：支持消息始终排第 1，但回答仍退化，说明检索命中不等于正确理解。
4. 展示 Finance/term_ambiguity_14 的两份 `Recorded judge output`，说明同一日期的表述可能收到不同评分；页面保留原始标签，没有自行改分。

页面包含完整已归档问答、两个 top-10 ID 列表及核对过的支持/干扰消息摘录；它不是全部十条消息正文的替代档案。原始运行路径与完整证据获取方式见 [Part 3 复现说明](../part3-exploration.md)。演示病例经过事后挑选，不代表总体准确率。查看参考答案不会改变检索结果。

交付前的小组动作：

- 通读报告并补成员与贡献；确认每位成员能解释 [审核表中的代码入口](../coursework-audit.md#代码面试与复现入口)。
- 独立核对四条 dev F9、一个角色修复、一个群组/主题修复，以及演示中的日期判分疑点。以姓名、日期、题目/方法、证据 ID 和判断记录复核，不覆盖正式模型标签。当前 AI 辅助审核不能填写为同学已完成的人工复核。
- 从报告三张图、RRF 公式和一正一负案例制作海报；本目录尚未生成最终海报。
- 与组员完成分支集成；在目标服务器按现有协议先做小样本验收，再决定是否需要重跑。此次报告与离线演示没有新模型调用。

目前不需要继续堆叠回复/时间规则来争取单次 test 分数。若没有具体正确性问题或新增教师要求，优先完成上述课程交付。
