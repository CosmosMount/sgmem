"""Build report figures and table rows from saved evidence; no model calls."""

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
CATEGORIES = ["multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity", "abstention"]
LABELS = ["Multi-hop", "Knowledge update", "Temporal", "User-implicit", "Term ambiguity", "Abstention"]
METHODS = ["bm25", "dense", "hybrid", "hybrid-meta"]


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def table_command(name, rows):
    return "\\newcommand{\\" + name + "}{%\n" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n}\n"


def build(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = read("part12-full-20261105/metrics.json")
    final = read("structured-test-comparison.json")
    dev = read("structured-qa-dev.json")
    failures = read("part12-full-20261105/failure-analysis.json")
    metrics = {m: baseline[m]["by_partition"]["test"] for m in METHODS[:2]}
    metrics.update({m: final["runs"][m]["metrics"] for m in METHODS[2:]})
    pairs = final["questions"]
    if len(pairs) != 536 or len({(r['domain'], r['category'], r['question_id']) for r in pairs}) != 536:
        raise ValueError("Expected the 536 unique frozen test questions")
    for method in METHODS:
        if metrics[method]["overall"]["total"] != 536:
            raise ValueError(f"Mismatched test denominator: {method}")
    for method in METHODS[2:]:
        if sum(r[method] for r in pairs) != metrics[method]["overall"]["correct"]:
            raise ValueError(f"Saved metrics disagree with paired outcomes: {method}")
    primary = Counter(r["primary_failure_type"] for r in [
        json.loads(line) for line in (EVIDENCE / "part12-full-20261105/failure-annotations.jsonl").read_text(encoding="utf-8").splitlines()
    ])
    if dict(primary) != failures["primary_failure_counts"] or sum(primary.values()) != 38:
        raise ValueError("Failure taxonomy disagrees with the saved annotations")

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#31476b", "#8794a5", "#c68c47", "#168b82"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.45), gridspec_kw={"width_ratios": [1, 1.2]}, layout="constrained")
    values = [100 * metrics[m]["overall"]["accuracy"] for m in METHODS]
    bars = axes[0].bar(["BM25", "Dense", "Hybrid", "Hybrid\n+ metadata"], values, color=colors, width=.65)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set(ylabel="Test accuracy (%)", ylim=(0, 40), title="(a) Same 536 test questions")
    deltas = [final["paired"]["by_category"][c]["net"] for c in CATEGORIES]
    bars = axes[1].barh(LABELS, deltas, color=[colors[3] if d >= 0 else "#ba564a" for d in deltas])
    axes[1].bar_label(bars, labels=[f"{d:+d}" for d in deltas], padding=3)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#555555", linewidth=.7)
    axes[1].set(xlabel="Net additional correct answers", xlim=(-2, 8), title="(b) Metadata versus hybrid")
    fig.savefig(output / "test-results.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    names = ["F1\nSpeaker", "F2\nGroup", "F3\nBelief", "F4\nReply", "F5\nUpdate", "F6\nCross-party", "F7\nRetrieval", "F8\nQA", "F9\nJudge review"]
    counts = [primary[f"F{i}"] for i in range(1, 10)]
    fig, ax = plt.subplots(figsize=(7.0, 1.95), layout="constrained")
    bars = ax.bar(names, counts, color=[colors[3]] * 6 + [colors[0]] * 2 + [colors[1]])
    ax.bar_label(bars, padding=2)
    ax.set(ylabel="Failure records", ylim=(0, 13), title="Development diagnosis: 38 labeled failures from 24 questions")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(output / "failure-taxonomy.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    variants = ["bm25-fields", "bm25-scope", "bm25-person", "bm25-time", "hybrid", "hybrid-meta", "bm25-scope-reply", "bm25-focus"]
    labels = ["Fields", "Scope", "Person", "Time", "Hybrid", "Hybrid-meta", "Scope+reply", "Focus"]
    offsets = [(5, -12), (5, -13), (-32, 7), (-26, 8), (5, 6), (5, 7), (-68, -11), (-29, 7)]
    fig, ax = plt.subplots(figsize=(7.0, 2.4), layout="constrained")
    for method, label, offset in zip(variants, labels, offsets):
        cost = dev["runs"][method]["cost"]["gpu_hours"]
        accuracy = 100 * dev["metrics"][method]["overall"]["accuracy"]
        ax.scatter(cost, accuracy, color=colors[3] if method == "hybrid-meta" else colors[0], s=38)
        ax.annotate(label, (cost, accuracy), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(xlabel="Measured GPU-hours for 179 dev questions", ylabel="Dev accuracy (%)", xlim=(.39, .94), ylim=(29.5, 37.5))
    ax.grid(alpha=.16)
    fig.savefig(output / "dev-cost.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    test_rows = []
    for category, label in zip(CATEGORIES, LABELS):
        row = [label, str(metrics["bm25"]["by_category"][category]["total"])]
        row += [f"{100 * metrics[m]['by_category'][category]['accuracy']:.2f}" for m in METHODS]
        test_rows.append(row)
    test_rows.append([r"\textbf{Overall}", "536", *[f"{v:.2f}" for v in values]])
    dev_rows = [[label, str(dev["metrics"][m]["overall"]["correct"]),
                 f"{100 * dev['metrics'][m]['overall']['accuracy']:.2f}",
                 f"{dev['runs'][m]['cost']['gpu_hours']:.3f}"] for m, label in zip(variants, labels)]
    round_hours = sum(json.loads(line)["wall_seconds"] / 3600
                      for name in ("structured-budget.jsonl", "structured-continuation-budget.jsonl")
                      for line in (EVIDENCE / name).read_text(encoding="utf-8").splitlines())
    receipts = read("historical-resource-receipts.json")["receipts"]
    extra_hours = sum(r["value"] / 3600 if r["unit"] == "seconds" else r["value"] for r in receipts)
    costs = [read("part12-full-20261105/provenance.json")["gpu_hours"],
             read("bounded-reply-qa-dev.json")["cost"]["gpu_hours"],
             read("ordered-reply-qa-dev.json")["cost"]["gpu_hours"], round_hours, extra_hours]
    cost_names = ["Full BM25/Dense baseline", "Bounded-reply dev", "Ordered-reply dev", "Metadata exploration and frozen test", "Additional recorded pilots and smoke tests"]
    cost_rows = [[name, f"{hours:.4f}"] for name, hours in zip(cost_names, costs)]
    cost_rows.append([r"\textbf{Recorded subtotal (incomplete history)}", f"{sum(costs):.4f}"])
    tables = table_command("TestRows", test_rows) + table_command("DevRows", dev_rows) + table_command("CostRows", cost_rows)
    (output.parent / "results.tex").write_text(tables, encoding="utf-8")
    print(json.dumps({"test_correct": {m: metrics[m]["overall"]["correct"] for m in METHODS},
                      "failure_records": sum(primary.values()), "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/report/figures")
    build(parser.parse_args().output)
