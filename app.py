import json
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

from benchmark.representations import REPRESENTATIONS
from benchmark.serialization import SERIALIZATION_MODES
from benchmark.search import INDEX_TYPES, RETRIEVAL_STRATEGIES
from benchmark.runner import run_experiment, run_grid, load_all_results
from benchmark.datasets import DATASETS
from benchmark.processing_runner import (
    clear_processing_results,
    get_processing_choices,
    get_processing_defaults,
    get_processing_run,
    load_all_processing_runs,
    request_processing_run_stop,
    set_processing_run_binned,
    start_processing_run,
)

app = Flask(__name__)


@app.route("/")
def index():
    results = load_all_results()
    return render_template(
        "index.html",
        datasets=list(DATASETS.keys()),
        representations=list(REPRESENTATIONS.keys()),
        serialization_modes=SERIALIZATION_MODES,
        index_types=INDEX_TYPES,
        retrievals=RETRIEVAL_STRATEGIES,
        results=results,
    )


@app.route("/run", methods=["POST"])
def run():
    dataset = request.form.get("dataset", "abt-buy")
    repr_names = request.form.getlist("representations")
    index_types = request.form.getlist("index_types")
    retrievals = request.form.getlist("retrievals")
    k_values = [int(x.strip()) for x in request.form.get("k_values", "5,10,20").split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in request.form.get("thresholds", "0.3,0.5,0.7").split(",") if x.strip()]
    serialization_mode = request.form.get("serialization_mode", "values_space")

    if not repr_names or not index_types or not retrievals:
        return redirect(url_for("index"))

    results = run_grid(
        dataset,
        repr_names,
        index_types,
        retrievals,
        k_values,
        thresholds,
        serialization_mode=serialization_mode,
    )
    return redirect(url_for("results_page"))


@app.route("/run-single", methods=["POST"])
def run_single():
    data = request.get_json()
    result = run_experiment(
        dataset_name=data.get("dataset", "abt-buy"),
        representation_name=data.get("representation", "tfidf"),
        index_type=data.get("index_type", "sklearn"),
        retrieval=data.get("retrieval", "topk"),
        k=int(data.get("k", 10)),
        threshold=float(data.get("threshold", 0.5)),
        serialization_mode=data.get("serialization_mode", "values_space"),
    )
    return jsonify(result)


@app.route("/results")
def results_page():
    results = load_all_results()
    return render_template("results.html", results=results)


@app.route("/processing-benchmark")
def processing_benchmark():
    defaults = get_processing_defaults()
    choices = get_processing_choices()
    all_runs = load_all_processing_runs()
    runs = [run for run in all_runs if not run.get("is_binned")]
    binned_runs = [run for run in all_runs if run.get("is_binned")]
    return render_template(
        "processing_benchmark.html",
        datasets=list(DATASETS.keys()),
        defaults=defaults,
        representations=choices["representations"],
        index_types=choices["index_types"],
        retrievals=choices["retrievals"],
        models=choices["models"],
        serialization_modes=choices["serialization_modes"],
        batching_modes=choices["batching_modes"],
        runs=runs,
        binned_runs=binned_runs,
    )


@app.route("/processing-benchmark/run", methods=["POST"])
def processing_benchmark_run():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        state = start_processing_run(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "run_id": state["run_id"],
            "detail_url": url_for("processing_benchmark_detail", run_id=state["run_id"]),
        }
    )


@app.route("/processing-benchmark/runs/<run_id>")
def processing_benchmark_detail(run_id):
    run = get_processing_run(run_id)
    if not run:
        return "Processing benchmark run not found", 404
    return render_template("processing_detail.html", run=run)


@app.route("/api/processing-benchmark/runs")
def api_processing_runs():
    return jsonify(load_all_processing_runs())


@app.route("/api/processing-benchmark/runs/<run_id>")
def api_processing_run(run_id):
    run = get_processing_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(run)


@app.route("/processing-benchmark/runs/<run_id>/bin", methods=["POST"])
def bin_processing_run(run_id):
    try:
        set_processing_run_binned(run_id, True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return redirect(url_for("processing_benchmark"))


@app.route("/processing-benchmark/runs/<run_id>/restore", methods=["POST"])
def restore_processing_run(run_id):
    try:
        set_processing_run_binned(run_id, False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return redirect(url_for("processing_benchmark"))


@app.route("/processing-benchmark/runs/<run_id>/stop", methods=["POST"])
def stop_processing_run(run_id):
    try:
        request_processing_run_stop(run_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    next_url = request.form.get("next") or request.args.get("next")
    if next_url == "detail":
        return redirect(url_for("processing_benchmark_detail", run_id=run_id))
    return redirect(url_for("processing_benchmark"))


@app.route("/processing-benchmark/clear-results", methods=["POST"])
def clear_processing_runs():
    clear_processing_results()
    return redirect(url_for("processing_benchmark"))


@app.route("/results/<experiment_id>")
def detail(experiment_id):
    path = os.path.join("results", f"{experiment_id}.json")
    if not os.path.exists(path):
        return "Experiment not found", 404
    with open(path) as f:
        result = json.load(f)
    return render_template("detail.html", result=result)


@app.route("/api/results")
def api_results():
    return jsonify(load_all_results())


@app.route("/clear-results", methods=["POST"])
def clear_results():
    import glob
    for f in glob.glob(os.path.join("results", "*.json")):
        os.remove(f)
    return redirect(url_for("results_page"))


if __name__ == "__main__":
    app.run(debug=True, port=5002, use_reloader=False)
