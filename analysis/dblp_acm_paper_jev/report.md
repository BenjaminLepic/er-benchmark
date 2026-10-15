# DBLP–ACM: Jev on the paper's official test pairs

Jev 1.13 evaluated the structured DBLP–ACM test split from the official archive for
[Can Foundation Models Wrangle Your Data?](https://arxiv.org/pdf/2205.09911).

| Evaluation | Pairs | Positives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Full test split | 2473 | 444 | 95.06% | 99.77% | 97.36% |
| First 200 test pairs | 200 | 39 | 97.50% | 100.00% | 98.73% |

Full split: 443 true positives, 23 false positives,
1 false negatives, and 2006 true negatives, according to supplied labels.
First 200: 39 true positives, 1 false positives,
0 false negatives, and 160 true negatives.

## Protocol

- Pinned model: jev-1.13.0, fixed Noul decision threshold 0.5, no demonstrations.
- Supplied test pairs are classified directly, with candidate generation bypassed.
- Attributes: title, authors, venue, and year; values_multiline serialization.
- Existing benchmark prompt adapted to publication records; no dataset-specific tuning.
- One-to-many native requests, up to ten candidate questions, concurrency five.
- 1275 requests, 915,512 input tokens, 45.063 seconds.
- Estimated cost: $0.038452, using repository pricing.

## Comparison with Table 1

The paper reports DBLP–ACM F1 of 93.5 for zero-shot GPT-3, 96.6 for GPT-3
with ten curated demonstrations, and 98.99 for Ditto.
Jev's first-200 score is 2.13 percentage points above the reported
few-shot GPT-3 value. This is provisional: the released script requests 200 pairs,
but correspondence with the exact Table 1 subset has not been independently verified.
The full-test score covers a different evaluation scope. Jev's native typed decisions
and existing benchmark prompt do not reproduce GPT-3 prompting.

## Artifacts and verification

predictions.csv contains all labels, model scores, predictions, and correctness flags.
errors.csv joins all 24 official-label errors to the original publication fields.
metrics.json records configuration, exact prompt, input file hashes, and usage.
Every test pair returned a score. Saved predictions reproduce both reported metric sets.
Inputs have unique record IDs, unique pairs, binary labels, and valid references.
All five existing TypeSafe integration tests passed.

Reproduce from the workspace root:

```sh
venv/bin/python -m analysis.run_amazon_google_paper --dataset DBLP-ACM
```

Rerunning makes fresh API requests and overwrites these DBLP–ACM artifacts.
The previous Amazon–Google artifacts remain available separately.
