# Entity Resolution Benchmark

A local Flask application for comparing candidate generation and model-based matching in entity resolution. Explore record representations, serialization formats, search indexes, retrieval strategies, and downstream matching models across product and publication datasets.

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
