#!/usr/bin/env python3
"""Generate a compact, reproducible report from saved blocking experiments.

The report intentionally focuses on comparable, dataset-agnostic sweeps:
labels_inline serialization, sklearn exact search, and corpus-size-derived k.
It uses only the Python standard library so report generation does not add a
plotting dependency to the benchmark runtime.
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.runner import load_all_results  # noqa: E402


OUT = ROOT / "analysis"
FIGURES = OUT / "figures"
DATASETS = [
    "abt-buy",
    "amazon-google",
    "dblp-acm",
    "dblp-scholar",
    "wdc-block-small",
]
DATASET_LABELS = {
    "abt-buy": "Abt–Buy",
    "amazon-google": "Amazon–Google",
    "dblp-acm": "DBLP–ACM",
    "dblp-scholar": "DBLP–Scholar",
    "wdc-block-small": "WDC Small",
}
REPRESENTATIONS = [
    "bm25-adaptive",
    "openai-small-512",
    "openai-small-1536",
    "gemini-768",
]
REP_LABELS = {
    "bm25-adaptive": "Adaptive BM25",
    "openai-small-512": "OpenAI small 512d",
    "openai-small-1536": "OpenAI small 1536d",
    "gemini-768": "Gemini 768d",
}
COLORS = {
    "Adaptive BM25 / top-k": "#64748b",
    "Adaptive BM25 / reciprocal": "#94a3b8",
    "OpenAI small 512d / top-k": "#2563eb",
    "OpenAI small 512d / reciprocal": "#60a5fa",
    "OpenAI small 1536d / top-k": "#7c3aed",
    "OpenAI small 1536d / reciprocal": "#a78bfa",
    "Gemini 768d / top-k": "#059669",
    "Gemini 768d / reciprocal": "#34d399",
    "top-k": "#059669",
    "reciprocal": "#2563eb",
    "stddev": "#d97706",
    "knee": "#7c3aed",
    "dropoff": "#dc2626",
    "auto_topk": "#059669",
    "auto_reciprocal": "#2563eb",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def dedupe_latest(results: list[dict]) -> list[dict]:
    """Defensive dedupe for direct JSON inputs; runner already does this."""
    latest = {}
    for result in results:
        key = (
            result.get("dataset"),
            result.get("serialization_mode"),
            result.get("representation"),
            result.get("index_type"),
            result.get("retrieval"),
            result.get("k"),
            result.get("threshold"),
            json.dumps(result.get("index_params", {}), sort_keys=True),
        )
        latest[key] = result
    return list(latest.values())


def expected_k(n_candidates: int, multiplier: int) -> int:
    return min(50, multiplier * math.ceil(math.log2(max(2, n_candidates))))


def core_sweep(results: list[dict]) -> list[dict]:
    rows = []
    for representation in REPRESENTATIONS:
        for retrieval in ("topk", "reciprocal"):
            for multiplier in range(1, 5):
                matches = []
                for result in results:
                    if not (
                        result.get("dataset") in DATASETS
                        and result.get("serialization_mode") == "labels_inline"
                        and result.get("representation") == representation
                        and result.get("index_type") == "sklearn"
                        and result.get("retrieval") == retrieval
                    ):
                        continue
                    if result.get("k") == expected_k(result["n_records_b"], multiplier):
                        matches.append(result)
                by_dataset = {result["dataset"]: result for result in matches}
                if set(by_dataset) != set(DATASETS):
                    continue
                ordered = [by_dataset[name] for name in DATASETS]
                rows.append(
                    {
                        "representation": representation,
                        "retrieval": retrieval,
                        "multiplier": multiplier,
                        "min_recall": min(r["recall"] for r in ordered),
                        "macro_recall": sum(r["recall"] for r in ordered) / len(ordered),
                        "mean_avg_k": sum(r["avg_k"] for r in ordered) / len(ordered),
                        "max_avg_k": max(r["avg_k"] for r in ordered),
                    }
                )
    return rows


def automatic_results(results: list[dict]) -> list[dict]:
    selected = [
        r
        for r in results
        if r.get("dataset") in DATASETS
        and r.get("serialization_mode") == "labels_inline"
        and r.get("representation") == "gemini-768"
        and r.get("index_type") == "sklearn"
        and r.get("retrieval") in ("auto_topk", "auto_reciprocal")
    ]
    return sorted(
        selected,
        key=lambda r: (DATASETS.index(r["dataset"]), r["retrieval"]),
    )


def wdc_sweep(results: list[dict]) -> list[dict]:
    selected = [
        r
        for r in results
        if r.get("dataset") == "wdc-block-small"
        and r.get("serialization_mode") == "labels_inline"
        and r.get("representation") == "gemini-768"
        and r.get("index_type") == "sklearn"
        and r.get("retrieval") in ("topk", "reciprocal", "dropoff", "knee", "stddev")
        and r.get("k") in {13, 26, 39, 50}
    ]
    return sorted(selected, key=lambda r: (r["retrieval"], r["avg_k"]))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def svg_line_chart(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_domain: tuple[float, float] | None = None,
    y_domain: tuple[float, float] | None = None,
) -> None:
    width, height = 980, 570
    left, right, top, bottom = 92, 265, 70, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for points in series.values() for point in points]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = x_domain or (min(xs), max(xs))
    ymin, ymax = y_domain or (min(ys), max(ys))
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,system-ui,-apple-system,sans-serif;fill:#243244}.title{font-size:22px;font-weight:700}.axis{font-size:13px}.legend{font-size:12px}.note{font-size:11px;fill:#64748b}</style>',
        f'<text x="{left}" y="34" class="title">{html.escape(title)}</text>',
    ]
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{x:.0f}</text>')
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{py + 4:.1f}" text-anchor="end" class="axis">{y:.3f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" class="axis">{html.escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + plot_w + 26
    for index, (name, points) in enumerate(series.items()):
        color = COLORS.get(name, "#334155")
        ordered = sorted(points)
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        dash = ' stroke-dasharray="7 5"' if "reciprocal" in name else ""
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        for x, y in ordered:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"><title>{html.escape(name)}: x={x:.2f}, y={y:.4f}</title></circle>')
        legend_y = top + 16 + index * 34
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{legend_x + 34}" y="{legend_y + 4}" class="legend">{html.escape(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def svg_grouped_bars(path: Path, rows: list[dict]) -> None:
    width, height = 980, 530
    left, right, top, bottom = 88, 42, 66, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.90, 1.0
    groups = defaultdict(dict)
    for row in rows:
        groups[row["dataset"]][row["retrieval"]] = row
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,system-ui,-apple-system,sans-serif;fill:#243244}.title{font-size:22px;font-weight:700}.axis{font-size:13px}.value{font-size:12px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text x="{left}" y="34" class="title">Automatic Gemini strategy recall by dataset</text>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = top + (ymax - value) / (ymax - ymin) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    slot = plot_w / len(DATASETS)
    bar_width = 42
    for group_index, dataset in enumerate(DATASETS):
        center = left + slot * (group_index + 0.5)
        for offset, retrieval in ((-24, "auto_topk"), (24, "auto_reciprocal")):
            row = groups[dataset][retrieval]
            value = row["recall"]
            y = top + (ymax - value) / (ymax - ymin) * plot_h
            base = top + plot_h
            color = COLORS[retrieval]
            x = center + offset - bar_width / 2
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{base - y:.1f}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{center + offset:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 28}" text-anchor="middle" class="axis">{html.escape(DATASET_LABELS[dataset])}</text>')
    legend_y = height - 25
    parts.append(f'<rect x="{left + 245}" y="{legend_y - 12}" width="14" height="14" fill="{COLORS["auto_topk"]}"/><text x="{left + 267}" y="{legend_y}" class="legend">auto top-k</text>')
    parts.append(f'<rect x="{left + 405}" y="{legend_y - 12}" width="14" height="14" fill="{COLORS["auto_reciprocal"]}"/><text x="{left + 427}" y="{legend_y}" class="legend">auto reciprocal</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(results: list[dict], frontier: list[dict], auto: list[dict], wdc: list[dict]) -> str:
    auto_by = {(r["dataset"], r["retrieval"]): r for r in auto}
    topk = [auto_by[(dataset, "auto_topk")] for dataset in DATASETS]
    reciprocal = [auto_by[(dataset, "auto_reciprocal")] for dataset in DATASETS]
    topk_min = min(r["recall"] for r in topk)
    topk_macro = sum(r["recall"] for r in topk) / len(topk)
    topk_avg = sum(r["avg_k"] for r in topk) / len(topk)
    reciprocal_min = min(r["recall"] for r in reciprocal)
    reciprocal_avg = sum(r["avg_k"] for r in reciprocal) / len(reciprocal)

    auto_rows = []
    for dataset in DATASETS:
        a = auto_by[(dataset, "auto_topk")]
        b = auto_by[(dataset, "auto_reciprocal")]
        auto_rows.append(
            [
                DATASET_LABELS[dataset],
                str(a["k"]),
                fmt(a["recall"]),
                fmt(b["recall"]),
                fmt(b["avg_k"], 1),
            ]
        )

    candidate_defaults = sorted(frontier, key=lambda row: (-row["min_recall"], row["mean_avg_k"]))
    frontier_rows = [
        [
            REP_LABELS[row["representation"]],
            row["retrieval"],
            f'{row["multiplier"]}×',
            fmt(row["min_recall"]),
            fmt(row["macro_recall"]),
            fmt(row["mean_avg_k"], 1),
        ]
        for row in candidate_defaults[:10]
    ]
    raw_count = len(list((ROOT / "results").glob("*.json")))
    return f"""# Dataset-Agnostic Blocking for Entity Resolution

## Experiment report

Generated {date.today().isoformat()} from **{raw_count:,} saved result files** ({len(results):,} unique configurations after deduplication).

### Abstract

This study asks whether a single candidate-generation policy can work across heterogeneous entity-resolution datasets without tuning thresholds for each dataset. We compare lexical and semantic representations, fixed and reciprocal nearest-neighbor retrieval, corpus-size-derived values of *k*, and score-shape heuristics. The strongest bounded default is Gemini-768 with inline field labels and

`k = min(50, 2 × ceil(log2(|target table|)))`.

It achieves **{topk_min:.3f} worst-dataset recall**, **{topk_macro:.3f} macro recall**, and **{topk_avg:.1f} candidates per query on average**. Reciprocal retrieval reduces the average output to **{reciprocal_avg:.1f} candidates** with a lower worst-dataset recall of **{reciprocal_min:.3f}**.

### 1. Experimental question

The objective is not to find a different optimum for every benchmark. It is to identify a generally usable blocking policy whose only adaptation comes from observable, label-free properties such as target-table size. Experiments use five datasets: Abt–Buy, Amazon–Google, DBLP–ACM, DBLP–Scholar, and WDC Products Block Small.

For the comparable core sweep:

- records are serialized with inline attribute labels;
- exact cosine search is used, avoiding approximate-index confounding;
- *k* is varied as 1×–4× `ceil(log2(|target|))`, capped at 50;
- results are aggregated using worst-dataset recall, macro recall, and mean candidates per query.

### 2. Cross-dataset recall–candidate frontier

![Worst-dataset recall versus candidates](figures/general_frontier.svg)

The vertical axis is deliberately the **minimum recall across all five datasets**, rather than an average that can hide failure on WDC. Gemini provides the strongest low-candidate frontier. OpenAI-1536 only slightly improves upon OpenAI-512, while increasing embedding time and storage. Adaptive BM25 remains a credible API-free fallback, but needs a larger candidate pool.

Top high-recall configurations:

{markdown_table(["Representation", "Retrieval", "k multiplier", "Min recall", "Macro recall", "Mean candidates"], frontier_rows)}

### 3. Automatic policy by dataset

![Automatic strategy recall by dataset](figures/automatic_recall_by_dataset.svg)

{markdown_table(["Dataset", "Resolved k", "Top-k recall", "Reciprocal recall", "Reciprocal avg k"], auto_rows)}

The legacy benchmarks are nearly saturated by both policies. WDC is the binding case and therefore determines the general-policy recall floor.

### 4. What WDC reveals

![WDC recall curves](figures/wdc_recall_vs_candidates.svg)

![WDC F1 curves](figures/wdc_f1_vs_candidates.svg)

WDC exposes a real precision–recall tradeoff that is mostly invisible on the older benchmarks. Reciprocal retrieval produces a smaller, higher-F1 candidate set. Top-k is preferable when downstream matching can tolerate approximately 26 candidates per query and blocking recall is the primary constraint.

The knee, dropoff, and score-standard-deviation methods can be compact, but their behavior is not stable across datasets. In particular, dropoff is attractive on several legacy benchmarks and loses too many WDC matches. Score-shape heuristics should therefore be presented as compact modes, not as the high-recall general default.

### 5. Metric interpretation

Blocking recall answers the operational question: *what fraction of true matches survive candidate generation?* Reduction ratio and average candidates quantify downstream work.

Pair precision and F1 require more care. The four legacy datasets use complete gold match mappings and the closed-world assumption that every other cross-table pair is a nonmatch. WDC's official split labels only a sampled pair universe, so its precision and F1 are computed within that sample. Consequently, WDC F1 should be compared **between WDC runs**, not directly with legacy-dataset F1.

### 6. Recommendation

Use **Gemini-768 + inline labels + automatic top-k at multiplier 2** as the bounded high-recall default. Expose automatic reciprocal retrieval as a lower-volume mode. If avoiding an external embedding API is important, use adaptive BM25 with multiplier 4. Do not make a similarity threshold the default until it has demonstrated a stable cross-dataset recall floor.

The next experiment should combine Gemini and adaptive BM25 ranks using a fixed reciprocal-rank-fusion rule. This tests whether lexical evidence can recover difficult WDC product matches without dataset-specific fitting.

### 7. Limitations

- Five datasets cover products and bibliographic records, but not people, organizations, addresses, or multilingual entities.
- WDC has a sampled labeled evaluation universe rather than exhaustive pair labels.
- API latency and pricing are operational considerations, not intrinsic blocking-quality measurements.
- The current Gemini representation embeds both tables with the same retrieval task type; asymmetric query/document embeddings remain worth testing.

### Reproduce this report

```bash
. venv/bin/activate
python analysis/generate_blocking_report.py
```

The generated CSV files contain the exact plotted aggregates and can be opened in a spreadsheet or statistical notebook.
"""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = dedupe_latest(load_all_results())
    frontier = core_sweep(results)
    auto = automatic_results(results)
    wdc = wdc_sweep(results)
    if len(frontier) != 32:
        raise RuntimeError(f"Expected 32 complete frontier configurations, found {len(frontier)}")
    if len(auto) != 10:
        raise RuntimeError(f"Expected 10 automatic results, found {len(auto)}")
    if len(wdc) != 20:
        raise RuntimeError(f"Expected 20 WDC sweep results, found {len(wdc)}")

    frontier_series = defaultdict(list)
    for row in frontier:
        name = f'{REP_LABELS[row["representation"]]} / {row["retrieval"].replace("topk", "top-k")}'
        frontier_series[name].append((row["mean_avg_k"], row["min_recall"]))
    svg_line_chart(
        FIGURES / "general_frontier.svg",
        dict(frontier_series),
        title="General-policy frontier across five datasets",
        x_label="Mean candidates per query",
        y_label="Worst-dataset blocking recall",
        x_domain=(5, 50),
        y_domain=(0.65, 1.0),
    )

    recall_series = defaultdict(list)
    f1_series = defaultdict(list)
    for row in wdc:
        name = row["retrieval"].replace("topk", "top-k")
        recall_series[name].append((row["avg_k"], row["recall"]))
        f1_series[name].append((row["avg_k"], row["f1"]))
    svg_line_chart(
        FIGURES / "wdc_recall_vs_candidates.svg",
        dict(recall_series),
        title="WDC Small: recall versus retained candidates",
        x_label="Average candidates per query",
        y_label="Blocking recall",
        x_domain=(0, 52),
        y_domain=(0.15, 1.0),
    )
    svg_line_chart(
        FIGURES / "wdc_f1_vs_candidates.svg",
        dict(f1_series),
        title="WDC Small: labeled-pair F1 versus retained candidates",
        x_label="Average candidates per query",
        y_label="F1 on the labeled pair universe",
        x_domain=(0, 52),
        y_domain=(0.25, 0.58),
    )
    svg_grouped_bars(FIGURES / "automatic_recall_by_dataset.svg", auto)

    write_csv(
        OUT / "core_frontier.csv",
        frontier,
        ["representation", "retrieval", "multiplier", "min_recall", "macro_recall", "mean_avg_k", "max_avg_k"],
    )
    write_csv(
        OUT / "automatic_results.csv",
        auto,
        ["dataset", "representation", "retrieval", "k", "avg_k", "recall", "precision", "f1", "reduction_ratio", "n_candidates"],
    )
    write_csv(
        OUT / "wdc_gemini_sweep.csv",
        wdc,
        ["retrieval", "k", "avg_k", "recall", "precision", "f1", "reduction_ratio", "n_candidates"],
    )
    (OUT / "blocking_experiment_report.md").write_text(build_report(results, frontier, auto, wdc))
    print(f"Wrote report: {OUT / 'blocking_experiment_report.md'}")
    print(f"Wrote figures: {FIGURES}")


if __name__ == "__main__":
    main()
