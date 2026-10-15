# Entity Resolution Benchmark

A local Flask application for comparing candidate generation and model-based matching in entity resolution. Explore record representations, serialization formats, search indexes, retrieval strategies, and downstream matching models across product and publication datasets.

## Recommended setup: FAISS + Jev

Our recommended starting point for a balance of accuracy, latency, and cost is **dense embeddings → FAISS Flat top-10 candidates → Jev 1.13 matching**. FAISS selects plausible matches; Jev decides which candidates represent the same entity. This is a practical recommendation based on the experiments below, rather than a claim of the highest accuracy on every dataset.

| Setting | Value |
|---|---|
| Embeddings | `openai-small-1536` (`text-embedding-3-small`, 1,536 dimensions) |
| Search index | `faiss_flat` (exact search with normalized inner product) |
| Retrieval | `topk`, `k = 10` |
| Matching model | `jev-1.13.0` through TypeSafe |
| Matching serialization | `values_multiline` |
| Batching | `one_to_many`, up to 10 candidates per source record |
| Concurrency | 20 requests, as used in the Abt–Buy runs; reduce for provider limits |
| Match decision | Jev `noul >= 0.5` |
| Prompt | Repository default; no custom instructions or demonstrations |

The processing runner builds embedding text with `values_space`; the matching serialization above controls the records sent to Jev. Its form field named **Threshold** controls retrieval strategies that filter on similarity. With `topk`, it does not filter candidates or change Jev's fixed 0.5 decision threshold. Set the table's values explicitly: the app's initial form defaults use a different configuration.

### Start a run

Follow the installation steps below and set `OPENAI_API_KEY` and `TYPESAFE_API_KEY` in your local `.env`. Open **Processing Benchmark** at http://127.0.0.1:5002/processing-benchmark and select the settings above, or submit this request while the app is running:

```sh
curl --fail-with-body http://127.0.0.1:5002/processing-benchmark/run \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "abt-buy",
    "representation": "openai-small-1536",
    "index_type": "faiss_flat",
    "retrieval": "topk",
    "k": 10,
    "threshold": 0.5,
    "serialization_mode": "values_multiline",
    "model": "jev-1.13.0",
    "batching_mode": "one_to_many",
    "batch_size": 10,
    "concurrency": 20,
    "custom_instructions": ""
  }'
```

This starts a background run and returns a `run_id` and `detail_url`. Open the detail URL on the local server to follow progress and inspect precision, recall, F1, latency, and estimated matching cost. Runs make paid API requests. The first run also generates embeddings; subsequent runs reuse the local embedding cache.

### Why we recommend it

On **Abt–Buy**, the [three saved Jev runs](analysis/jev_benchmark_report.md) used the embedding, top-10, prompt, and matching settings above with **sklearn** as the search backend. They achieved mean **93.87% F1**, mean **94.63% precision**, mean **93.13% recall**, and median **13.656 seconds**, at **$0.114906 estimated matching cost** per run. Candidate generation retained **98.9061%** of known matches in **10,810 pairs**, compared with 1,180,452 possible pairs.

Those timings used cached embeddings, and the estimated cost covers Jev matching, excluding embedding charges. The saved comparison shows Jev close to the strongest observed model F1 at substantially lower matching cost and runtime; it does not establish Jev as the highest-accuracy model. See the [individual run configuration and metrics](results/processing/9b01d0c9.json).

FAISS Flat is our proposed exact-search backend for this recipe. The historical end-to-end figures above were recorded with sklearn and should not be labeled as measured FAISS + Jev performance. Approximate indexes (`faiss_hnsw`, `faiss_ivf`) require separate recall checks before substituting them.

A local check on October 15, 2026 reused the cached Abt–Buy embeddings and compared `faiss_flat` with sklearn at top-10. Both produced 10,810 candidates and retrieved the same 1,085 of 1,097 known matches (98.9061% recall). Five candidate pairs were replaced across five source records. This verifies candidate recall for that input; Jev was not rerun on the changed candidate set, so final FAISS + Jev F1 remains unmeasured.

Choose candidate count for the data: matches missed during retrieval cannot be recovered by Jev. The [five-dataset blocking study](analysis/blocking_experiment_report.md) supports exploring Gemini-768 and `auto_topk` (`min(50, 2 × ceil(log2(target-table size)))`) when higher candidate recall matters. That study uses a separate serialization and evaluation protocol; its results do not measure the combined Jev pipeline.

Accuracy also varies by domain: Jev scored **97.36% F1 on DBLP–ACM** and **68.54% on Amazon–Google** when classifying the supplied paper test pairs directly. Those runs bypass retrieval. See the [DBLP–ACM report](analysis/dblp_acm_paper_jev/report.md) and [Amazon–Google report](analysis/amazon_google_paper_jev/report.md) before generalizing the recommendation to a new dataset.

## Run locally

```sh
git clone https://github.com/BenjaminLepic/er-benchmark.git
cd er-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open http://127.0.0.1:5002. The application runs with Flask's development server and debug mode enabled; use it locally.

TF-IDF and BM25 experiments can run without API credentials. For hosted embeddings or model-based matching, set the relevant provider key in `.env`. Gemini also requires the optional `google-genai` package (`pip install google-genai`). Local sentence-transformer representations download model weights on first use. PostgreSQL search options require a configured database with pgvector.

## Project contents

- `app.py`, `templates/`, `static/`: experiment interface and result views.
- `benchmark/`: dataset loading, representations, retrieval, evaluation, and matching runners.
- `data/`: benchmark CSVs for Abt–Buy, Amazon–Google, DBLP–ACM, DBLP–Scholar, WDC Products, and paper-comparison splits. Third-party datasets retain their original terms.
- `results/`: saved experiment results used by the analysis scripts.
- `analysis/`: reports, figures, comparison outputs, and scripts.
- `tests/`: unit tests for TypeSafe processing requests and scoring.

Embedding caches, virtual environments, and local credentials are excluded from Git. New experiment runs write results into `results/`.

## Reports

- [Dataset-agnostic blocking experiments](analysis/blocking_experiment_report.md)
- [Jev benchmark comparison](analysis/jev_benchmark_report.md)
- [Amazon–Google paper comparison](analysis/amazon_google_paper_jev/report.md)
- [DBLP–ACM paper comparison](analysis/dblp_acm_paper_jev/report.md)

## Tests

```sh
python -m unittest discover -s tests -v
```
