# Amazon–Google: Jev compared with the paper

Run completed on 2026-10-12 using the official dataset archive from
[Can Foundation Models Wrangle Your Data?](https://arxiv.org/pdf/2205.09911).

## Results

All metrics below are percentages. Jev uses no task demonstrations.

| Evaluation | Pairs | Positives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Jev 1.13, full test split | 2,293 | 234 | 54.07 | 93.59 | 68.54 |
| Jev 1.13, first 200 test pairs | 200 | 22 | 58.33 | 95.45 | 72.41 |
| Paper GPT-3, zero-shot (Table 1) | — | — | — | — | 54.3 |
| Paper GPT-3, ten curated demonstrations (Table 1) | — | — | — | — | 63.5 |
| Paper Ditto (Table 1) | — | — | — | — | 75.58 |

The first-200 result is 8.91 F1 percentage points above the paper's reported
ten-shot GPT-3 score. This is a provisional comparison: the released experiment
script requests 200 examples, and its inference code processes the first 200
merged test rows, but correspondence with the exact Table 1 evaluation subset
has not been independently verified. The full-test result has a different
evaluation scope and should not be treated as a matched comparison with Table 1.

## Protocol

- Data: official `datasets/entity_matching/structured/Amazon-Google` archive files.
- Evaluate supplied positive and negative test pairs directly, bypassing retrieval.
- Model: pinned `jev-1.13.0`, native TypeSafe Noul questions.
- Fixed decision threshold: 0.5; no threshold tuning or training/validation labels used.
- Existing benchmark prompt and `values_multiline` serialization.
- One source record with up to ten independent candidate questions per request.
- Concurrency: 5; requests: 793.
- Runtime: 26.191 seconds; input tokens: 671,094.
- Estimated cost: $0.028186, using the repository's configured pricing.

The paper uses GPT-3 text prompting and curated demonstrations. This run uses
Jev's native typed decisions and the existing benchmark prompt; it does not
reproduce the GPT-3 prompting implementation.

## Errors and verification

Full split: 219 true positives, 186 false positives, 15 false negatives, and
1,873 true negatives. First 200: 21 true positives, 15 false positives, one false
negative, and 163 true negatives. False positives are the main source of error.

Input checks verified unique record IDs, unique test pairs, binary labels, and
valid record references. Every test pair returned a score. A metric sanity check
verified the confusion counts and exclusion of predictions outside the selected
evaluation universe. All five existing TypeSafe integration tests passed.

## Artifacts and reproduction

- `predictions.csv`: all test labels, Jev scores, decisions, and correctness flags
  in original test-file order.
- `metrics.json`: metrics, configuration, exact prompt, usage, cost, and SHA-256
  fingerprints for all three input files.
- Input splits are saved under `data/fm-paper/Amazon-Google`.

From the workspace root, rerun with:

```sh
venv/bin/python -m analysis.run_amazon_google_paper
```

A rerun makes fresh API requests and overwrites these output artifacts.

References: [official repository](https://github.com/HazyResearch/fm_data_tasks),
[experiment script](https://github.com/HazyResearch/fm_data_tasks/blob/main/scripts/run_results.zsh),
[inference implementation](https://github.com/HazyResearch/fm_data_tasks/blob/main/fm_data_tasks/run_inference.py).
