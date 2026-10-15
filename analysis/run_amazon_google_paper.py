"""Evaluate Jev on the paper's supplied pairs, without retrieval.

Run from the workspace: venv/bin/python -m analysis.run_amazon_google_paper
Extract the official archive's datasets/entity_matching/structured/Amazon-Google
directory to data/fm-paper/Amazon-Google first.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import pandas as pd

from benchmark.evaluate import evaluate
from benchmark.processing_models import DEFAULT_BASE_PROMPT, estimate_usage_cost
from benchmark.processing_prompts import build_system_prompt
from benchmark.processing_runner import (
    _build_batches,
    _build_typesafe_batch_request,
    _execute_processing_requests_async,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fm-paper/Amazon-Google"
MODEL = "jev-1.13.0"


def load_inputs(data=DATA):
    tables = [pd.read_csv(data / name, dtype={"id": str})
              for name in ("tableA.csv", "tableB.csv")]
    labels = pd.read_csv(data / "test.csv", dtype={"ltable_id": str, "rtable_id": str})
    assert set(labels.label) <= {0, 1}, "Unexpected labels"
    records = []
    for table in tables:
        assert table.id.is_unique, "Duplicate record IDs"
        fields = [c for c in table.columns if c != "id" and not c.startswith("Unnamed")]
        records.append({row["id"]: {c: "" if pd.isna(row[c]) else str(row[c])
                                   for c in fields}
                        for row in table.to_dict("records")})
    pairs = list(zip(labels.ltable_id, labels.rtable_id))
    assert len(set(pairs)) == len(pairs), "Duplicate test pairs"
    assert all(a in records[0] and b in records[1] for a, b in pairs), "Missing records"
    return tables, records, labels, pairs


def score_pairs(labels, predictions):
    universe = set(zip(labels.ltable_id, labels.rtable_id))
    positives = labels[labels.label == 1]
    gold = set(zip(positives.ltable_id, positives.rtable_id))
    predicted = predictions & universe
    metrics = evaluate(predicted, gold, labels.ltable_id.nunique(),
                       labels.rtable_id.nunique(), labeled_pairs=universe)
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return {k: metrics[k] for k in ("precision", "recall", "f1")} | {
        "pairs": len(universe), "positives": len(gold),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": len(universe) - tp - fp - fn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["Amazon-Google", "DBLP-ACM"], default="Amazon-Google")
    args = parser.parse_args()
    data = ROOT / "data/fm-paper" / args.dataset
    output_name, entity_type = {
        "Amazon-Google": ("amazon_google_paper_jev", "product records"),
        "DBLP-ACM": ("dblp_acm_paper_jev", "publication"),
    }[args.dataset]
    tables, records, labels, pairs = load_inputs(data)
    system_prompt = build_system_prompt(DEFAULT_BASE_PROMPT, entity_type, "")
    batches = _build_batches(pairs, "one_to_many", 10)
    bundles = []
    for i, batch in enumerate(batches):
        request = _build_typesafe_batch_request(
            batch, records[0], records[1], "values_multiline", MODEL, system_prompt)
        bundles.append(request | {"sequence": i, "pairs": batch["pairs"], "mode": batch["mode"]})
    print(f"Running {MODEL}: {len(pairs)} test pairs, {int(labels.label.sum())} positives, "
          f"{len(batches)} requests, threshold 0.5, concurrency 5", flush=True)
    started = time.perf_counter()
    result = asyncio.run(_execute_processing_requests_async(
        "paper-" + uuid.uuid4().hex[:8], MODEL, bundles, {"recall": 1.0}, len(pairs), 5))
    elapsed = time.perf_counter() - started
    scores = result["decision_scores"]
    assert set(scores) == set(pairs), "Missing or unexpected model scores"
    predictions = result["predicted_pairs"]
    output = ROOT / "analysis" / output_name
    output.mkdir(exist_ok=True)
    scored = labels.copy()
    scored["jev_score"] = [scores[p] for p in pairs]
    scored["prediction"] = (scored.jev_score >= 0.5).astype(int)
    scored["correct"] = scored.prediction == scored.label
    scored.to_csv(output / "predictions.csv", index=False)
    cost = estimate_usage_cost(MODEL, result["total_input_tokens"],
                              result["total_output_tokens"], request_count=len(batches))
    summary = {
        "dataset": args.dataset,
        "model": MODEL, "threshold": 0.5, "demonstrations": 0,
        "serialization": "values_multiline", "batching": "one_to_many",
        "batch_size": 10, "concurrency": 5, "system_prompt": system_prompt,
        "dataset_url": "https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz",
        "file_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                        for name in ("tableA.csv", "tableB.csv", "test.csv")},
        "full_test": score_pairs(labels, predictions),
        "first_200": score_pairs(labels.iloc[:200], predictions),
        "seconds": round(elapsed, 3), "requests": len(batches),
        "input_tokens": result["total_input_tokens"], "cost": cost,
        "comparison_note": "First 200 test rows follow released script selection; "
                           "paper Table 1 subset correspondence is not independently verified. "
                           "Jev uses native typed decisions, the existing benchmark prompt, "
                           "and no demonstrations; this is not a replication of GPT-3 prompting.",
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
