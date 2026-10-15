# Jev 1.13 entity-resolution benchmark

## Setup

- Dataset: Abt-Buy
- Candidate generation: `openai-small-1536` / sklearn / top-k 10
- Candidate pairs: 10,810, with candidate recall 0.989061
- Serialization: `values_multiline`
- Decision rule: Jev `noul >= 0.5`
- Jev model: pinned `jev-1.13.0`
- Native Jev batching: one source record and all 10 candidates in shared state, with one independent Noul question per candidate

This differs from a text-generation integration: Jev receives structured state and typed yes/no questions and returns duplicate probabilities directly. There is no generated JSON decision to parse.

## Jev results

Three native batch-size-10 repetitions were run:

| Run | Precision | Recall | F1 | Cost | Time |
|---|---:|---:|---:|---:|---:|
| `9b01d0c9` | 0.947173 | 0.931632 | 0.939338 | $0.114906 | 14.072 s |
| `ec4d2594` | 0.944547 | 0.931632 | 0.938045 | $0.114906 | 13.656 s |
| `393944b2` | 0.947124 | 0.930720 | 0.938851 | $0.114906 | 13.610 s |

Mean F1 was 0.938745 (sample standard deviation 0.000653). Mean precision was 0.946281, mean recall was 0.931328, and median time was 13.656 seconds.

The same provider integration at batch size 5 (`5c746656`) produced F1 0.930023 at $0.131830 in 26.817 seconds. Native batch size 10 is both better and cheaper here.

## Comparison with existing runs

The table below holds the dataset, candidate set, serialization, and decision task constant, and uses the best observed F1 for models with multiple runs. Jev uses its provider-native batch size of 10; the text-generation models shown use batch size 5. Times reflect each run's configured concurrency and should be treated as end-to-end observations, not isolated per-request latency.

| Model | Precision | Recall | F1 | Cost | Time |
|---|---:|---:|---:|---:|---:|
| Grok 4.1 Fast Reasoning | 0.967526 | 0.923428 | 0.944963 | $0.365651 | 355.372 s |
| GPT-5.6 Luna | 0.926056 | 0.958979 | 0.942230 | $0.854085 | 85.031 s |
| Jev 1.13 | 0.947173 | 0.931632 | 0.939338 | $0.114906 | 14.072 s |
| GPT-5.4 Mini | 0.941828 | 0.929809 | 0.935780 | $3.331531 | 207.587 s |
| Grok 4.1 Fast non-reasoning | 0.925960 | 0.923428 | 0.924692 | $0.353498 | 28.235 s |
| DeepSeek Reasoner | 0.972222 | 0.829535 | 0.895229 | $7.997172 | 346.549 s |

On the best Jev run, Jev was 0.289 percentage points below the best Luna F1 while costing 86.5% less, and 0.563 points below the best Grok reasoning F1 while costing 68.6% less. It beat the comparable GPT-5.4 Mini run by 0.356 points while costing 96.6% less.

An in-sample threshold diagnostic found a best Jev F1 of 0.940799 at threshold 0.48 for run `9b01d0c9`, only 0.146 points above the default 0.5 result. This is diagnostic only; the 0.5 result is the fair reported benchmark because 0.48 was selected on the evaluation labels.
