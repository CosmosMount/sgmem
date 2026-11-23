"""Render saved frozen-test cases as an offline HTML demonstration. No inference."""

import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/structured-test-cases.json"


def recorded_text(value):
    # Keep original line breaks without trailing whitespace in generated HTML.
    return escape(value).replace("\r", "&#13;").replace("\n", "&#10;")


def render_case(case):
    label = f"{case['domain']}/{case['question_id']}"
    cells = []
    for method in ("hybrid", "hybrid-meta"):
        row = case["methods"][method]
        status = "Correct" if row["official_correct"] else "Incorrect"
        cells.append(f"<article><h3>{escape(method)} <span class='{status.lower()}'>Recorded judge: {status}</span></h3>"
                     f"<p>{recorded_text(row['answer'])}</p><details><summary>Recorded top-10 message IDs</summary>"
                     f"<ol>{''.join('<li>'+escape(mid)+'</li>' for mid in row['retrieved_ids'])}</ol></details>"
                     f"<details><summary>Recorded judge output</summary><p>{recorded_text(row['judge_output'])}</p></details></article>")
    sources = []
    for message in case["source_messages"]:
        ranks = ", ".join(f"{m}: {case['methods'][m]['source_message_ranks'][message['id']] or 'outside top-10'}"
                          for m in ("hybrid", "hybrid-meta"))
        meta = " | ".join(str(message.get(k) or "unknown") for k in ("speaker", "role", "group", "timestamp"))
        sources.append(f"<details><summary>{escape(message['id'])} - {escape(ranks)}</summary>"
                       f"<p class='meta'>{escape(meta)}</p><p>{recorded_text(message['content'])}</p></details>")
    return (f"<section><h2>{escape(label)}</h2><p class='meta'>{escape(case['diagnostic_kind'])}</p>"
            f"<p><b>Question:</b> {recorded_text(case['question'])}</p>"
            f"<details><summary>Reference answer (evaluation only)</summary><p>{recorded_text(case['reference_answer'])}</p></details>"
            f"<div class='pair'>{''.join(cells)}</div><h3>Checked source messages</h3>{''.join(sources)}</section>")


def build(output, selected=None):
    evidence = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = evidence["cases"]
    if selected:
        cases = [c for c in cases if f"{c['domain']}/{c['question_id']}" == selected]
        if not cases:
            raise ValueError(f"Unknown saved case: {selected}. Use --list to see available cases.")
    body = "".join(render_case(case) for case in cases)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SGMem - Recorded case replay</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#243248;background:#f2f5f7;margin:0}
main{max-width:1100px;margin:auto;padding:28px}h1,h2,h3{line-height:1.25}h1{margin-bottom:8px}
.notice{background:#e0eee9;padding:16px;border-left:4px solid #168b82}.meta{color:#59677b;font-size:14px}
section{background:white;margin-top:24px;padding:24px;border-radius:10px;border:1px solid #dde3e9}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}article{background:#f5f7f9;padding:16px}
span{display:inline-block;font-size:13px;padding:3px 8px;border-radius:4px}.correct{background:#d2ece1}.incorrect{background:#f4ddd8}
details{margin:10px 0;border-top:1px solid #dce3e9;padding-top:10px}summary{cursor:pointer;font-weight:600}
p{white-space:pre-wrap;overflow-wrap:anywhere}ol{font-size:14px}footer{padding-top:22px;color:#59677b}
@media(max-width:760px){main{padding:14px}.pair{grid-template-columns:1fr}section{padding:16px}}
</style><main><h1>SGMem: recorded case replay</h1>
<p>Frozen test comparison: hybrid and hybrid-meta</p>
<div class="notice"><b>Offline replay, not live inference.</b> Answers, rankings and judge labels below are saved experimental outputs.
Saved post-hoc cases illustrate successes and failures; they are not a representative accuracy sample.
Reference answers are shown for evaluation only and were not used during retrieval.</div>
""" + body + """<footer>Source: docs/evidence/structured-test-cases.json. Conversation and question excerpts from
<a href="https://github.com/UCSB-NLP-Chang/GroupMemBench/tree/461f00c3e8a1a4a3a2ed5814b729014502c4702f">GroupMemBench</a>
by Yang et al., licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
Excerpts are selected and presented with our model outputs; source wording is unchanged.
No GPU or network access is needed to view the replay.</footer></main></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the six saved case IDs")
    parser.add_argument("--case", help="Show only DOMAIN/QUESTION_ID")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/report/case-demo.html")
    args = parser.parse_args()
    if args.list:
        for case in json.loads(SOURCE.read_text(encoding="utf-8"))["cases"]:
            print(f"{case['domain']}/{case['question_id']}: {case['diagnostic_kind']}")
    else:
        try:
            print(f"Replayed {build(args.output, args.case)} saved cases to {args.output}")
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
