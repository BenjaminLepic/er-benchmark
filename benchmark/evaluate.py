def evaluate(candidate_pairs, ground_truth, n_source_a, n_source_b, labeled_pairs=None):
    """
    Compute blocking evaluation metrics.

    Args:
        candidate_pairs: set of (id_a, id_b) candidate pairs
        ground_truth: set of (id_a, id_b) true matching pairs
        n_source_a: number of records in source A
        n_source_b: number of records in source B
        labeled_pairs: optional set defining the pairs for which positive/negative
            labels are complete. Candidate count and reduction ratio still use the
            full search space; precision and F1 use this labeled universe.

    Returns:
        dict with recall, reduction_ratio, precision, f1
    """
    total_possible = n_source_a * n_source_b
    true_positives = len(candidate_pairs & ground_truth)
    n_candidates = len(candidate_pairs)
    n_ground_truth = len(ground_truth)

    if labeled_pairs is None:
        evaluated_candidates = candidate_pairs
        evaluation_mode = "complete_ground_truth"
    else:
        evaluated_candidates = candidate_pairs & labeled_pairs
        evaluation_mode = "labeled_pairs"

    recall = true_positives / n_ground_truth if n_ground_truth > 0 else 0.0
    reduction_ratio = 1.0 - (n_candidates / total_possible) if total_possible > 0 else 0.0
    n_evaluated_candidates = len(evaluated_candidates)
    precision = true_positives / n_evaluated_candidates if n_evaluated_candidates > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    avg_k = round(n_candidates / n_source_a, 2) if n_source_a > 0 else 0.0

    return {
        "recall": round(recall, 6),
        "reduction_ratio": round(reduction_ratio, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "true_positives": true_positives,
        "n_candidates": n_candidates,
        "n_evaluated_candidates": n_evaluated_candidates,
        "n_ground_truth": n_ground_truth,
        "total_possible_pairs": total_possible,
        "avg_k": avg_k,
        "evaluation_mode": evaluation_mode,
    }
