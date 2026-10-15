import json
import os
import time
import uuid

import numpy as np
from scipy import sparse

from benchmark.datasets import load_dataset_with_evaluation
from benchmark.representations import REPRESENTATIONS, SPARSE_REPRESENTATIONS
from benchmark.search import (
    run_search, INDEX_TYPES, RETRIEVAL_STRATEGIES,
    _cross_file_indices, build_index, _to_dense,
    retrieve_topk, retrieve_threshold, retrieve_topk_threshold,
    retrieve_reciprocal, retrieve_dropoff, retrieve_knee, retrieve_stddev,
    resolve_retrieval,
)
from benchmark.evaluate import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_experiment(dataset_name, representation_name, index_type, retrieval,
                   k=10, threshold=0.5, serialization_mode="values_space", **index_params):
    """Run a single blocking experiment."""
    experiment_id = str(uuid.uuid4())[:8]

    df, ground_truth, (source_a, source_b), labeled_pairs = load_dataset_with_evaluation(
        dataset_name,
        serialization_mode=serialization_mode,
    )
    ids = df["id"].tolist()
    sources = df["source"].tolist()
    texts = df["text"].tolist()
    n_a = sum(1 for s in sources if s == source_a)
    n_b = sum(1 for s in sources if s == source_b)

    repr_fn = REPRESENTATIONS[representation_name]
    t0 = time.time()
    vectors, repr_label = repr_fn(texts)
    repr_time = time.time() - t0

    _, effective_k = resolve_retrieval(retrieval, k, n_b, index_params)

    candidate_pairs, build_time, query_time = run_search(
        index_type, retrieval, vectors, ids, sources, source_a, source_b,
        k=k, threshold=threshold, **index_params,
    )

    metrics = evaluate(candidate_pairs, ground_truth, n_a, n_b, labeled_pairs=labeled_pairs)

    result = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "serialization_mode": serialization_mode,
        "representation": representation_name,
        "repr_label": repr_label,
        "index_type": index_type,
        "retrieval": retrieval,
        "k": effective_k,
        "threshold": threshold,
        "index_params": index_params,
        "n_records_a": n_a,
        "n_records_b": n_b,
        "repr_time_s": round(repr_time, 3),
        "index_build_time_s": round(build_time, 3),
        "query_time_s": round(query_time, 3),
        "total_time_s": round(repr_time + build_time + query_time, 3),
        **metrics,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_grid(dataset_name, representation_names, index_types, retrievals,
             k_values, thresholds=None, serialization_mode="values_space", **index_params):
    """Run a grid of experiments. Returns list of result dicts."""
    if thresholds is None:
        thresholds = [0.5]

    results = []

    df, ground_truth, (source_a, source_b), labeled_pairs = load_dataset_with_evaluation(
        dataset_name,
        serialization_mode=serialization_mode,
    )
    ids = df["id"].tolist()
    sources = df["source"].tolist()
    texts = df["text"].tolist()
    n_a = sum(1 for s in sources if s == source_a)
    n_b = sum(1 for s in sources if s == source_b)

    for repr_name in representation_names:
        repr_fn = REPRESENTATIONS[repr_name]
        t0 = time.time()
        vectors, repr_label = repr_fn(texts)
        repr_time = time.time() - t0

        for idx_type in index_types:
            # Skip non-sklearn indexes for sparse representations (high-dim, sparse)
            if repr_name in SPARSE_REPRESENTATIONS and idx_type != "sklearn":
                continue
            # pgvector_hnsw has 2000 dim limit
            if idx_type == "pgvector_hnsw" and repr_name in (
                "openai-large-3072", "kalm-gemma-3840", "nemotron-4096",
            ):
                continue

            idx_a, idx_b = _cross_file_indices(ids, sources, source_a, source_b)
            if sparse.issparse(vectors):
                vecs_b = vectors[idx_b]
            else:
                vecs_b = np.asarray(vectors[idx_b], dtype=np.float32)

            ids_b = [ids[i] for i in idx_b]

            # Build index once per (representation, index_type)
            t_build = time.time()
            if idx_type == "sklearn":
                index_obj, _ = build_index(idx_type, vecs_b, ids_b, **index_params)
            else:
                index_obj, _ = build_index(idx_type, _to_dense(vecs_b), ids_b, **index_params)
            build_time = time.time() - t_build

            if sparse.issparse(vectors):
                vecs_a = vectors[idx_a]
            else:
                vecs_a = np.asarray(vectors[idx_a], dtype=np.float32)

            for retrieval in retrievals:
                needs_k = retrieval in ("topk", "topk_threshold", "reciprocal", "dropoff", "knee", "stddev")
                needs_threshold = retrieval in ("threshold", "topk_threshold")

                if needs_k and needs_threshold:
                    combos = [(kv, t) for kv in k_values for t in thresholds]
                elif needs_k:
                    combos = [(kv, None) for kv in k_values]
                elif needs_threshold:
                    combos = [(None, t) for t in thresholds]
                else:
                    combos = [(None, None)]

                for k_val, t_val in combos:
                    experiment_id = str(uuid.uuid4())[:8]
                    resolved_retrieval, effective_k = resolve_retrieval(
                        retrieval, k_val, len(idx_b), index_params
                    )

                    dispatch = {
                        "topk": lambda: retrieve_topk(index_obj, vecs_a, ids, idx_a, idx_b, k=effective_k),
                        "threshold": lambda: retrieve_threshold(index_obj, vecs_a, ids, idx_a, idx_b, threshold=t_val),
                        "topk_threshold": lambda: retrieve_topk_threshold(index_obj, vecs_a, ids, idx_a, idx_b, k=effective_k, threshold=t_val),
                        "reciprocal": lambda: retrieve_reciprocal(index_obj, vecs_a, vecs_b, ids, idx_a, idx_b, k=effective_k),
                        "dropoff": lambda: retrieve_dropoff(index_obj, vecs_a, ids, idx_a, idx_b, k=effective_k),
                        "knee": lambda: retrieve_knee(index_obj, vecs_a, ids, idx_a, idx_b, k=effective_k),
                        "stddev": lambda: retrieve_stddev(index_obj, vecs_a, ids, idx_a, idx_b, k=effective_k),
                    }
                    pairs, query_time = dispatch[resolved_retrieval]()

                    metrics = evaluate(pairs, ground_truth, n_a, n_b, labeled_pairs=labeled_pairs)

                    result = {
                        "experiment_id": experiment_id,
                        "dataset": dataset_name,
                        "serialization_mode": serialization_mode,
                        "representation": repr_name,
                        "repr_label": repr_label,
                        "index_type": idx_type,
                        "retrieval": retrieval,
                        "k": effective_k,
                        "threshold": t_val,
                        "index_params": index_params,
                        "n_records_a": n_a,
                        "n_records_b": n_b,
                        "repr_time_s": round(repr_time, 3),
                        "index_build_time_s": round(build_time, 3),
                        "query_time_s": round(query_time, 3),
                        "total_time_s": round(repr_time + build_time + query_time, 3),
                        **metrics,
                    }

                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
                    with open(path, "w") as f:
                        json.dump(result, f, indent=2)

                    results.append(result)

    return results


def load_all_results():
    """Load all saved experiment results, deduplicated by config (keep newest)."""
    raw = []
    if not os.path.exists(RESULTS_DIR):
        return raw
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(RESULTS_DIR, fname)
            mtime = os.path.getmtime(path)
            with open(path) as f:
                r = json.load(f)
                r["_mtime"] = mtime
                raw.append(r)

    # Deduplicate: keep only the newest result per config
    best = {}
    for r in raw:
        key = (
            r.get("dataset"),
            r.get("serialization_mode", "values_space"),
            r.get("representation"),
            r.get("index_type"),
            r.get("retrieval"),
            r.get("k"),
            r.get("threshold"),
            json.dumps(r.get("index_params", {}), sort_keys=True),
        )
        if key not in best or r["_mtime"] > best[key]["_mtime"]:
            best[key] = r

    results = list(best.values())
    for r in results:
        r.pop("_mtime", None)
    return sorted(results, key=lambda r: r.get("total_time_s", 0))
