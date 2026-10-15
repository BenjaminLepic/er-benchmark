import json
import os
import threading
import time
import traceback
import uuid
import asyncio
from collections import defaultdict
from datetime import datetime

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

from benchmark.datasets import DATASET_SPECS, load_processing_dataset
from benchmark.evaluate import evaluate
from benchmark.processing_models import (
    BATCHING_MODES,
    DEFAULT_BASE_PROMPT,
    PROCESSING_MODELS,
    SERIALIZATION_MODES,
    estimate_usage_cost,
    get_model_provider,
)
from benchmark.processing_prompts import (
    MULTI_PAIR_SCHEMA,
    ONE_TO_MANY_SCHEMA,
    PAIRWISE_SCHEMA,
    build_multi_pair_prompt,
    build_one_to_many_prompt,
    build_pairwise_prompt,
    build_request,
    build_system_prompt,
    extract_response_text,
    extract_usage,
    parse_json_response,
    serialize_record,
)
from benchmark.representations import REPRESENTATIONS, SPARSE_REPRESENTATIONS
from benchmark.search import INDEX_TYPES, RETRIEVAL_STRATEGIES, run_search

load_dotenv()

PROCESSING_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results",
    "processing",
)

_RUNS = {}
_RUN_LOCK = threading.Lock()
_MAX_RETRIES = 5
_BASE_DELAY_S = 1.0
_OPENAI_CONNECT_TIMEOUT_S = 20.0
_OPENAI_READ_TIMEOUT_S = 180.0
_OPENAI_WRITE_TIMEOUT_S = 60.0
_OPENAI_POOL_TIMEOUT_S = 60.0
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_TYPESAFE_BASE_URL = "https://api.typesafe.ai"
_MAX_PARSE_RETRIES = 2


class RunStoppedError(RuntimeError):
    """Raised when a processing run is stopped by the user."""


def _utcnow_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _safe_int(value, default=None):
    if value is None or value == "":
        return default
    return int(value)


def get_processing_defaults():
    return {
        "dataset": "abt-buy",
        "representation": "tfidf",
        "index_type": "sklearn",
        "retrieval": "topk",
        "k": 1,
        "threshold": 0.5,
        "serialization_mode": "labels_inline",
        "model": "gpt-5-nano",
        "batching_mode": "pairwise",
        "batch_size": 5,
        "concurrency": 5,
        "base_prompt": DEFAULT_BASE_PROMPT,
        "custom_instructions": "",
    }


def get_processing_choices():
    return {
        "models": PROCESSING_MODELS,
        "serialization_modes": SERIALIZATION_MODES,
        "batching_modes": BATCHING_MODES,
        "index_types": INDEX_TYPES,
        "retrievals": RETRIEVAL_STRATEGIES,
        "representations": list(REPRESENTATIONS.keys()),
    }


def _update_run(run_id, **updates):
    with _RUN_LOCK:
        state = _RUNS.setdefault(run_id, {})
        state.update(updates)
        state["updated_at"] = _utcnow_iso()


def _get_run_state(run_id):
    with _RUN_LOCK:
        state = _RUNS.get(run_id)
        return dict(state) if state else None


def _is_stop_requested(run_id):
    with _RUN_LOCK:
        state = _RUNS.get(run_id, {})
        return bool(state.get("stop_requested"))


def _raise_if_stop_requested(run_id):
    if _is_stop_requested(run_id):
        raise RunStoppedError("Run stopped by user")


def _persist_result(run_id, result):
    os.makedirs(PROCESSING_RESULTS_DIR, exist_ok=True)
    path = os.path.join(PROCESSING_RESULTS_DIR, f"{run_id}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    return path


def load_processing_result(run_id):
    path = os.path.join(PROCESSING_RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _persist_processing_result(run_id, state):
    os.makedirs(PROCESSING_RESULTS_DIR, exist_ok=True)
    path = os.path.join(PROCESSING_RESULTS_DIR, f"{run_id}.json")
    with open(path, "w") as handle:
        json.dump(state, handle, indent=2)
    return path


def _sort_runs(runs):
    def sort_key(run):
        return (
            run.get("created_at", ""),
            run.get("run_id", ""),
        )

    return sorted(runs, key=sort_key, reverse=True)


def load_all_processing_runs():
    results = []
    if os.path.exists(PROCESSING_RESULTS_DIR):
        for filename in os.listdir(PROCESSING_RESULTS_DIR):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(PROCESSING_RESULTS_DIR, filename)) as handle:
                results.append(json.load(handle))

    persisted_ids = {result["run_id"] for result in results if "run_id" in result}
    with _RUN_LOCK:
        for run_id, state in _RUNS.items():
            if run_id not in persisted_ids:
                results.append(dict(state))

    return _sort_runs(results)


def get_processing_run(run_id):
    state = _get_run_state(run_id)
    if state:
        return state
    return load_processing_result(run_id)


def set_processing_run_binned(run_id, is_binned=True):
    state = load_processing_result(run_id)
    if state is None:
        raise ValueError("Run not found")

    if state.get("status") not in {"completed", "failed", "stopped"}:
        raise ValueError("Only completed, failed, or stopped runs can be moved to the bin")

    state["is_binned"] = bool(is_binned)
    if is_binned:
        state["binned_at"] = _utcnow_iso()
    else:
        state.pop("binned_at", None)

    _persist_processing_result(run_id, state)

    with _RUN_LOCK:
        if run_id in _RUNS:
            _RUNS[run_id]["is_binned"] = bool(is_binned)
            if is_binned:
                _RUNS[run_id]["binned_at"] = state["binned_at"]
            else:
                _RUNS[run_id].pop("binned_at", None)

    return state


def request_processing_run_stop(run_id):
    with _RUN_LOCK:
        state = _RUNS.get(run_id)
        if state is None:
            raise ValueError("Run not found or no longer running")

        status = state.get("status")
        if status not in {"queued", "running", "stopping"}:
            raise ValueError("Only queued or running runs can be stopped")

        state["stop_requested"] = True
        if status != "stopping":
            state["status"] = "stopping"
        progress = dict(state.get("progress") or {})
        if progress:
            progress["stage"] = "stopping"
            state["progress"] = progress
        state["updated_at"] = _utcnow_iso()
        return dict(state)


def clear_processing_results():
    with _RUN_LOCK:
        _RUNS.clear()
    if not os.path.exists(PROCESSING_RESULTS_DIR):
        return
    for filename in os.listdir(PROCESSING_RESULTS_DIR):
        if filename.endswith(".json"):
            os.remove(os.path.join(PROCESSING_RESULTS_DIR, filename))


def validate_processing_config(config):
    dataset_name = config["dataset"]
    representation_name = config["representation"]
    index_type = config["index_type"]
    retrieval = config["retrieval"]
    serialization_mode = config["serialization_mode"]
    batching_mode = config["batching_mode"]
    model_name = config["model"]

    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if representation_name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation_name}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type}")
    if retrieval not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {retrieval}")
    if serialization_mode not in SERIALIZATION_MODES:
        raise ValueError(f"Unknown serialization mode: {serialization_mode}")
    if batching_mode not in BATCHING_MODES:
        raise ValueError(f"Unknown batching mode: {batching_mode}")
    if model_name not in PROCESSING_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if config.get("batch_size", 1) < 1:
        raise ValueError("Batch size must be at least 1")
    if config.get("concurrency", 1) < 1:
        raise ValueError("Concurrency must be at least 1")
    if config.get("k", 1) < 1:
        raise ValueError("k must be at least 1")

    if representation_name in SPARSE_REPRESENTATIONS and index_type != "sklearn":
        raise ValueError("Sparse representations only support the sklearn index")

    if index_type == "pgvector_hnsw" and representation_name in (
        "openai-large-3072",
        "kalm-gemma-3840",
        "nemotron-4096",
    ):
        raise ValueError("This representation exceeds pgvector HNSW dimension limits")


def start_processing_run(raw_config):
    defaults = get_processing_defaults()
    config = {
        **defaults,
        **raw_config,
    }
    config["k"] = _safe_int(config.get("k"), default=defaults["k"])
    config["batch_size"] = _safe_int(config.get("batch_size"), default=defaults["batch_size"])
    config["concurrency"] = _safe_int(config.get("concurrency"), default=defaults["concurrency"])
    config["threshold"] = _safe_float(config.get("threshold"))
    config["base_prompt"] = config.get("base_prompt") or DEFAULT_BASE_PROMPT
    config["custom_instructions"] = config.get("custom_instructions") or ""

    validate_processing_config(config)

    run_id = str(uuid.uuid4())[:8]
    initial_state = {
        "run_id": run_id,
        "status": "queued",
        "is_binned": False,
        "stop_requested": False,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "dataset": config["dataset"],
        "blocking_config": {
            "representation": config["representation"],
            "index_type": config["index_type"],
            "retrieval": config["retrieval"],
            "k": config["k"],
            "threshold": config["threshold"],
        },
        "processing_config": {
            "serialization_mode": config["serialization_mode"],
            "model": config["model"],
            "batching_mode": config["batching_mode"],
            "batch_size": config["batch_size"],
            "concurrency": config["concurrency"],
        },
        "progress": {
            "stage": "queued",
            "completed_requests": 0,
            "total_requests": 0,
            "completed_pairs": 0,
            "total_pairs": 0,
        },
    }
    with _RUN_LOCK:
        _RUNS[run_id] = initial_state

    thread = threading.Thread(
        target=_run_processing_benchmark_background,
        args=(run_id, config),
        daemon=True,
    )
    thread.start()
    return initial_state


async def _call_with_retries_async(client, request_type, request_kwargs):
    delay = _BASE_DELAY_S
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            if request_type == "responses":
                return await client.responses.create(**request_kwargs)
            if request_type == "chat_completions":
                return await client.chat.completions.create(**request_kwargs)
            if request_type == "chat_completions_parse":
                return await client.beta.chat.completions.parse(**request_kwargs)
            if request_type == "google_generate_content":
                return await asyncio.to_thread(client.models.generate_content, **request_kwargs)
            if request_type == "typesafe_systemone":
                response = await client.post("/v1/systemone", **request_kwargs)
                response.raise_for_status()
                return response
            raise ValueError(f"Unsupported request type: {request_type}")
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        except APIError as exc:
            if not exc.status_code or exc.status_code < 500 or exc.status_code >= 600:
                raise
            last_error = exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code not in {429, 529} and not 500 <= status_code < 600:
                raise
            last_error = exc
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        except httpx.TransportError as exc:
            last_error = exc

        if attempt == _MAX_RETRIES - 1:
            break
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)

    if isinstance(last_error, APITimeoutError):
        raise RuntimeError(
            f"OpenAI request timed out after {_MAX_RETRIES} attempts. "
            "Try rerunning with lower concurrency or a smaller batch size."
        ) from last_error

    if isinstance(last_error, APIConnectionError):
        raise RuntimeError(
            f"OpenAI connection failed after {_MAX_RETRIES} attempts. "
            "Check network connectivity, then try again with lower concurrency."
        ) from last_error

    if isinstance(last_error, httpx.TransportError):
        raise RuntimeError(
            f"Provider connection failed after {_MAX_RETRIES} attempts. "
            "Check network connectivity, then try again with lower concurrency."
        ) from last_error

    raise last_error


def _build_candidate_pairs(dataset_bundle, config):
    df = dataset_bundle["combined_df"]
    ids = df["id"].tolist()
    sources = df["source"].tolist()
    texts = df["text"].tolist()

    repr_fn = REPRESENTATIONS[config["representation"]]
    repr_started = time.perf_counter()
    vectors, repr_label = repr_fn(texts)
    repr_time = time.perf_counter() - repr_started

    search_started = time.perf_counter()
    candidate_pairs, build_time, query_time = run_search(
        config["index_type"],
        config["retrieval"],
        vectors,
        ids,
        sources,
        dataset_bundle["source_a"],
        dataset_bundle["source_b"],
        k=config["k"],
        threshold=config["threshold"] if config["threshold"] is not None else 0.5,
    )
    search_time = time.perf_counter() - search_started

    return {
        "candidate_pairs": sorted(candidate_pairs),
        "repr_label": repr_label,
        "repr_time_s": round(repr_time, 3),
        "index_build_time_s": round(build_time, 3),
        "query_time_s": round(query_time, 3),
        "blocking_stage_time_s": round(search_time, 3),
    }


def _pair_id(index):
    return f"pair_{index}"


def _chunk(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _build_pair_debug_rows(pairs, dataset_bundle):
    source_a = dataset_bundle["source_a"]
    source_b = dataset_bundle["source_b"]
    records_a = dataset_bundle["records"][source_a]
    records_b = dataset_bundle["records"][source_b]

    rows = []
    for id_a, id_b in sorted(pairs):
        rows.append(
            {
                "record_a_id": id_a,
                "record_b_id": id_b,
                "record_a_fields": records_a.get(id_a, {}),
                "record_b_fields": records_b.get(id_b, {}),
            }
        )
    return rows


def _build_batches(candidate_pairs, batch_mode, batch_size):
    if batch_mode == "pairwise":
        return [
            {"mode": "pairwise", "pairs": [pair]}
            for pair in candidate_pairs
        ]

    if batch_mode == "multi_pair":
        return [
            {"mode": "multi_pair", "pairs": batch}
            for batch in _chunk(candidate_pairs, batch_size)
        ]

    if batch_mode == "one_to_many":
        grouped = defaultdict(list)
        for id_a, id_b in candidate_pairs:
            grouped[id_a].append(id_b)
        batches = []
        for id_a in sorted(grouped):
            for id_b_chunk in _chunk(sorted(grouped[id_a]), batch_size):
                batches.append(
                    {
                        "mode": "one_to_many",
                        "id_a": id_a,
                        "pairs": [(id_a, id_b) for id_b in id_b_chunk],
                    }
                )
        return batches

    raise ValueError(f"Unknown batching mode: {batch_mode}")


def _build_batch_request(batch, dataset_bundle, serialization_mode, model_name, system_prompt):
    source_a = dataset_bundle["source_a"]
    source_b = dataset_bundle["source_b"]
    records_a = dataset_bundle["records"][source_a]
    records_b = dataset_bundle["records"][source_b]

    if get_model_provider(model_name) == "typesafe":
        return _build_typesafe_batch_request(
            batch,
            records_a,
            records_b,
            serialization_mode,
            model_name,
            system_prompt,
        )

    if batch["mode"] == "pairwise":
        id_a, id_b = batch["pairs"][0]
        user_prompt = build_pairwise_prompt(
            serialize_record(records_a[id_a], serialization_mode),
            serialize_record(records_b[id_b], serialization_mode),
        )
        return {
            "request": build_request(model_name, system_prompt, user_prompt, PAIRWISE_SCHEMA),
            "user_prompt": user_prompt,
            "pair_lookup": {_pair_id(0): (id_a, id_b)},
            "candidate_lookup": {id_b: (id_a, id_b)},
        }

    if batch["mode"] == "multi_pair":
        prompt_pairs = []
        pair_lookup = {}
        for index, (id_a, id_b) in enumerate(batch["pairs"]):
            local_pair_id = _pair_id(index)
            pair_lookup[local_pair_id] = (id_a, id_b)
            prompt_pairs.append(
                {
                    "pair_id": local_pair_id,
                    "record_a": serialize_record(records_a[id_a], serialization_mode),
                    "record_b": serialize_record(records_b[id_b], serialization_mode),
                }
            )
        user_prompt = build_multi_pair_prompt(prompt_pairs)
        return {
            "request": build_request(model_name, system_prompt, user_prompt, MULTI_PAIR_SCHEMA),
            "user_prompt": user_prompt,
            "pair_lookup": pair_lookup,
            "candidate_lookup": {},
        }

    if batch["mode"] == "one_to_many":
        id_a = batch["id_a"]
        candidates = []
        candidate_lookup = {}
        for _, id_b in batch["pairs"]:
            candidate_lookup[id_b] = (id_a, id_b)
            candidates.append(
                {
                    "candidate_id": id_b,
                    "record_b": serialize_record(records_b[id_b], serialization_mode),
                }
            )
        user_prompt = build_one_to_many_prompt(
            serialize_record(records_a[id_a], serialization_mode),
            candidates,
        )
        return {
            "request": build_request(model_name, system_prompt, user_prompt, ONE_TO_MANY_SCHEMA),
            "user_prompt": user_prompt,
            "pair_lookup": {},
            "candidate_lookup": candidate_lookup,
        }

    raise ValueError(f"Unsupported batch mode: {batch['mode']}")


def _typesafe_guidance(system_prompt):
    output_suffix = "\n\nReturn only the structured decision output for the records you evaluate."
    if system_prompt.endswith(output_suffix):
        return system_prompt[:-len(output_suffix)]
    return system_prompt


def _typesafe_duplicate_question(system_prompt, target):
    return {
        "type": "noul",
        "instructions": {
            "question": f"Do the two records in {target} represent the same real-world entity?",
            "guidance": _typesafe_guidance(system_prompt),
        },
        "criteria": {
            "true": "They are the same entity; ordinary typos, abbreviations, and formatting differences are allowed.",
            "false": "They are different entities, especially when key identifiers or product identity conflict.",
        },
    }


def _build_typesafe_batch_request(
    batch,
    records_a,
    records_b,
    serialization_mode,
    model_name,
    system_prompt,
):
    pair_lookup = {}
    questions = {}

    if batch["mode"] == "pairwise":
        id_a, id_b = batch["pairs"][0]
        state = {
            "record_a": serialize_record(records_a[id_a], serialization_mode),
            "record_b": serialize_record(records_b[id_b], serialization_mode),
        }
        question_id = _pair_id(0)
        pair_lookup[question_id] = (id_a, id_b)
        questions[question_id] = _typesafe_duplicate_question(
            system_prompt,
            "record_a and record_b",
        )
    elif batch["mode"] == "multi_pair":
        candidate_pairs = []
        for index, (id_a, id_b) in enumerate(batch["pairs"]):
            question_id = _pair_id(index)
            pair_lookup[question_id] = (id_a, id_b)
            candidate_pairs.append(
                {
                    "pair_id": question_id,
                    "record_a": serialize_record(records_a[id_a], serialization_mode),
                    "record_b": serialize_record(records_b[id_b], serialization_mode),
                }
            )
            questions[question_id] = _typesafe_duplicate_question(
                system_prompt,
                f"candidate_pairs entry with pair_id {question_id}",
            )
        state = {"candidate_pairs": candidate_pairs}
    elif batch["mode"] == "one_to_many":
        id_a = batch["id_a"]
        candidates = []
        for index, (_, id_b) in enumerate(batch["pairs"]):
            question_id = _pair_id(index)
            pair_lookup[question_id] = (id_a, id_b)
            candidates.append(
                {
                    "candidate_id": question_id,
                    "record": serialize_record(records_b[id_b], serialization_mode),
                }
            )
            questions[question_id] = _typesafe_duplicate_question(
                system_prompt,
                f"source_record and candidates entry with candidate_id {question_id}",
            )
        state = {
            "source_record": serialize_record(records_a[id_a], serialization_mode),
            "candidates": candidates,
        }
    else:
        raise ValueError(f"Unsupported batch mode: {batch['mode']}")

    payload = {
        "state": state,
        "model": PROCESSING_MODELS[model_name]["api_model"],
        "questions": questions,
    }
    return {
        "request": {
            "request_type": "typesafe_systemone",
            "request_kwargs": {"json": payload},
        },
        "user_prompt": json.dumps(payload, ensure_ascii=False, indent=2),
        "pair_lookup": pair_lookup,
        "candidate_lookup": {},
    }


def _extract_batch_predictions(batch_mode, payload, pair_lookup, candidate_lookup):
    predicted_pairs = set()

    if "answers" in payload:
        threshold = 0.5
        for question_id, answer in payload.get("answers", {}).items():
            if answer.get("type") == "noul" and answer.get("noul", 0.0) >= threshold:
                pair = pair_lookup.get(question_id)
                if pair:
                    predicted_pairs.add(pair)
        return predicted_pairs

    if batch_mode == "pairwise":
        if payload.get("is_duplicate"):
            predicted_pairs.add(pair_lookup[_pair_id(0)])
        return predicted_pairs

    if batch_mode == "multi_pair":
        for decision in payload.get("decisions", []):
            pair_id = decision.get("pair_id")
            if decision.get("is_duplicate") and pair_id in pair_lookup:
                predicted_pairs.add(pair_lookup[pair_id])
        return predicted_pairs

    if batch_mode == "one_to_many":
        for decision in payload.get("decisions", []):
            candidate_id = decision.get("candidate_id")
            if decision.get("is_duplicate") and candidate_id in candidate_lookup:
                predicted_pairs.add(candidate_lookup[candidate_id])
        return predicted_pairs

    raise ValueError(f"Unsupported batch mode: {batch_mode}")


def _extract_typesafe_scores(payload, pair_lookup):
    scores = {}
    for question_id, answer in payload.get("answers", {}).items():
        pair = pair_lookup.get(question_id)
        if pair and answer.get("type") == "noul" and "noul" in answer:
            scores[pair] = float(answer["noul"])
    return scores


async def _execute_batches_async(run_id, client, request_bundles, blocking_metrics, total_pairs, concurrency):
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_input_tokens = 0
    total_cache_write_tokens = 0
    predicted_pairs = set()
    request_durations = []
    completed_requests = 0
    completed_pairs = 0
    response_text_by_sequence = {}
    decision_scores = {}

    async def execute_one(bundle):
        async with semaphore:
            _raise_if_stop_requested(run_id)
            last_parse_error = None
            total_request_duration = 0.0
            last_raw_response_text = ""
            for _ in range(_MAX_PARSE_RETRIES + 1):
                request_started = time.perf_counter()
                response = await _call_with_retries_async(
                    client,
                    bundle["request"]["request_type"],
                    bundle["request"]["request_kwargs"],
                )
                total_request_duration += time.perf_counter() - request_started
                try:
                    if bundle["request"]["request_type"] == "typesafe_systemone":
                        payload = response.json()
                        raw_response_text = json.dumps(payload, ensure_ascii=False)
                        usage = payload.get("usage", {})
                        normalized_usage = {
                            "input_tokens": int(usage.get("input_tokens", 0)),
                            "output_tokens": int(usage.get("output_tokens", 0)),
                            "cached_input_tokens": 0,
                            "cache_write_tokens": 0,
                        }
                    else:
                        payload, raw_response_text = parse_json_response(response)
                        normalized_usage = extract_usage(response)
                    return {
                        "sequence": bundle["sequence"],
                        "pair_count": len(bundle["pairs"]),
                        "duration": total_request_duration,
                        "payload": payload,
                        "response_text": raw_response_text,
                        "usage": normalized_usage,
                        "mode": bundle["mode"],
                        "pair_lookup": bundle["pair_lookup"],
                        "candidate_lookup": bundle["candidate_lookup"],
                        "decision_scores": _extract_typesafe_scores(
                            payload,
                            bundle["pair_lookup"],
                        ),
                    }
                except json.JSONDecodeError as exc:
                    last_parse_error = exc
                    try:
                        last_raw_response_text = extract_response_text(response)
                    except Exception:
                        last_raw_response_text = ""
                    await asyncio.sleep(0.5)

            preview = last_raw_response_text[:500] if last_raw_response_text else ""
            raise RuntimeError(
                f"Model returned invalid JSON after {_MAX_PARSE_RETRIES + 1} attempts: {last_parse_error}. "
                f"Response preview: {preview!r}"
            )

    tasks = [asyncio.create_task(execute_one(bundle)) for bundle in request_bundles]

    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            completed_requests += 1
            completed_pairs += result["pair_count"]
            request_durations.append(result["duration"])
            response_text_by_sequence[result["sequence"]] = result["response_text"]
            decision_scores.update(result["decision_scores"])

            predicted_pairs.update(
                _extract_batch_predictions(
                    result["mode"],
                    result["payload"],
                    result["pair_lookup"],
                    result["candidate_lookup"],
                )
            )

            usage = result["usage"]
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
            total_cached_input_tokens += usage["cached_input_tokens"]
            total_cache_write_tokens += usage.get("cache_write_tokens", 0)

            if _is_stop_requested(run_id):
                raise RunStoppedError("Run stopped by user")

            _update_run(
                run_id,
                status="running",
                progress={
                    "stage": "llm_evaluation",
                    "completed_requests": completed_requests,
                    "total_requests": len(request_bundles),
                    "completed_pairs": completed_pairs,
                    "total_pairs": total_pairs,
                    "candidate_pairs": total_pairs,
                    "candidate_recall": blocking_metrics["recall"],
                },
            )
    except RunStoppedError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_input_tokens": total_cached_input_tokens,
        "total_cache_write_tokens": total_cache_write_tokens,
        "predicted_pairs": predicted_pairs,
        "request_durations": request_durations,
        "first_response_text": response_text_by_sequence.get(0, ""),
        "decision_scores": decision_scores,
    }


def _run_processing_benchmark_background(run_id, config):
    try:
        _update_run(run_id, status="running", progress={"stage": "loading_dataset"})
        result = run_processing_benchmark(run_id, config)
        _persist_result(run_id, result)
        in_memory_result = dict(result)
        in_memory_result.pop("run_id", None)
        _update_run(run_id, **in_memory_result)
    except RunStoppedError:
        state = _get_run_state(run_id) or {}
        progress = dict(state.get("progress") or {})
        progress["stage"] = "stopped"
        stopped = {
            "run_id": run_id,
            "status": "stopped",
            "is_binned": False,
            "stop_requested": True,
            "created_at": state.get("created_at", _utcnow_iso()),
            "completed_at": _utcnow_iso(),
            "dataset": config.get("dataset"),
            "repr_label": state.get("repr_label"),
            "blocking_config": {
                "representation": config.get("representation"),
                "index_type": config.get("index_type"),
                "retrieval": config.get("retrieval"),
                "k": config.get("k"),
                "threshold": config.get("threshold"),
            },
            "processing_config": {
                "serialization_mode": config.get("serialization_mode"),
                "base_prompt": config.get("base_prompt"),
                "custom_instructions": config.get("custom_instructions"),
                "model": config.get("model"),
                "batching_mode": config.get("batching_mode"),
                "batch_size": config.get("batch_size"),
                "concurrency": config.get("concurrency"),
            },
            "blocking_metrics": state.get("blocking_metrics"),
            "counts": state.get("counts"),
            "progress": progress,
        }
        _persist_result(run_id, stopped)
        in_memory_stopped = dict(stopped)
        in_memory_stopped.pop("run_id", None)
        _update_run(run_id, **in_memory_stopped)
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "status": "failed",
            "is_binned": False,
            "created_at": _get_run_state(run_id).get("created_at", _utcnow_iso()),
            "completed_at": _utcnow_iso(),
            "dataset": config.get("dataset"),
            "blocking_config": {
                "representation": config.get("representation"),
                "index_type": config.get("index_type"),
                "retrieval": config.get("retrieval"),
                "k": config.get("k"),
                "threshold": config.get("threshold"),
            },
            "processing_config": {
                "serialization_mode": config.get("serialization_mode"),
                "base_prompt": config.get("base_prompt"),
                "custom_instructions": config.get("custom_instructions"),
                "model": config.get("model"),
                "batching_mode": config.get("batching_mode"),
                "batch_size": config.get("batch_size"),
                "concurrency": config.get("concurrency"),
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _persist_result(run_id, failure)
        in_memory_failure = dict(failure)
        in_memory_failure.pop("run_id", None)
        _update_run(run_id, **in_memory_failure)


def _build_provider_client(model_name, timeout):
    provider = get_model_provider(model_name)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run processing benchmarks")
        return {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 0,
        }

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required to run DeepSeek processing benchmarks")
        return {
            "api_key": api_key,
            "base_url": "https://api.deepseek.com",
            "timeout": timeout,
            "max_retries": 0,
        }

    if provider == "xai":
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is required to run xAI Grok processing benchmarks")
        return {
            "api_key": api_key,
            "base_url": "https://api.x.ai/v1",
            "timeout": timeout,
            "max_retries": 0,
        }

    if provider == "google":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required to run Gemini processing benchmarks")
        from google import genai
        return genai.Client(api_key=api_key)

    if provider == "typesafe":
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required to run TypeSafe processing benchmarks")
        return {
            "base_url": _TYPESAFE_BASE_URL,
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "timeout": timeout,
        }

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to run processing benchmarks with OpenRouter models")
        return {
            "api_key": api_key,
            "base_url": _OPENROUTER_BASE_URL,
            "default_headers": {
                "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost"),
                "X-Title": os.environ.get("OPENROUTER_X_TITLE", "er-benchmark"),
            },
            "timeout": timeout,
            "max_retries": 0,
        }

    raise RuntimeError(f"Unsupported processing model provider: {provider}")


async def _execute_processing_requests_async(run_id, model_name, request_bundles, blocking_metrics, total_pairs, concurrency):
    provider = get_model_provider(model_name)
    if provider == "google":
        client = _build_provider_client(model_name, None)
        return await _execute_batches_async(
            run_id,
            client,
            request_bundles,
            blocking_metrics,
            total_pairs,
            concurrency,
        )

    if provider == "typesafe":
        timeout = httpx.Timeout(
            connect=_OPENAI_CONNECT_TIMEOUT_S,
            read=_OPENAI_READ_TIMEOUT_S,
            write=_OPENAI_WRITE_TIMEOUT_S,
            pool=_OPENAI_POOL_TIMEOUT_S,
        )
        client_kwargs = _build_provider_client(model_name, timeout)
        async with httpx.AsyncClient(**client_kwargs) as client:
            return await _execute_batches_async(
                run_id,
                client,
                request_bundles,
                blocking_metrics,
                total_pairs,
                concurrency,
            )

    timeout = httpx.Timeout(
        connect=_OPENAI_CONNECT_TIMEOUT_S,
        read=_OPENAI_READ_TIMEOUT_S,
        write=_OPENAI_WRITE_TIMEOUT_S,
        pool=_OPENAI_POOL_TIMEOUT_S,
    )
    client_kwargs = _build_provider_client(model_name, timeout)
    async with AsyncOpenAI(**client_kwargs) as client:
        return await _execute_batches_async(
            run_id,
            client,
            request_bundles,
            blocking_metrics,
            total_pairs,
            concurrency,
        )


def run_processing_benchmark(run_id, config):
    started_at = time.perf_counter()
    created_at = _get_run_state(run_id).get("created_at", _utcnow_iso())

    dataset_bundle = load_processing_dataset(config["dataset"])
    _raise_if_stop_requested(run_id)
    candidate_result = _build_candidate_pairs(dataset_bundle, config)
    candidate_pairs = candidate_result["candidate_pairs"]

    blocking_metrics = evaluate(
        set(candidate_pairs),
        dataset_bundle["ground_truth"],
        dataset_bundle["n_records_a"],
        dataset_bundle["n_records_b"],
        labeled_pairs=dataset_bundle.get("labeled_pairs"),
    )

    batches = _build_batches(candidate_pairs, config["batching_mode"], config["batch_size"])
    total_requests = len(batches)
    total_pairs = len(candidate_pairs)

    _update_run(
        run_id,
        status="running",
        repr_label=candidate_result["repr_label"],
        blocking_metrics=blocking_metrics,
        counts={
            "n_records_a": dataset_bundle["n_records_a"],
            "n_records_b": dataset_bundle["n_records_b"],
            "n_ground_truth": len(dataset_bundle["ground_truth"]),
            "candidate_pairs": len(candidate_pairs),
            "avg_candidates_per_record": blocking_metrics["avg_k"],
            "predicted_matches": 0,
            "llm_requests": total_requests,
        },
        progress={
            "stage": "blocking_complete",
            "completed_requests": 0,
            "total_requests": total_requests,
            "completed_pairs": 0,
            "total_pairs": total_pairs,
            "candidate_pairs": total_pairs,
            "candidate_recall": blocking_metrics["recall"],
        },
    )
    _raise_if_stop_requested(run_id)

    system_prompt = build_system_prompt(
        config["base_prompt"],
        dataset_bundle["entity_type"],
        config["custom_instructions"],
    )

    llm_started = time.perf_counter()
    first_user_prompt = ""
    request_bundles = []

    for batch_index, batch in enumerate(batches):
        _raise_if_stop_requested(run_id)
        request_bundle = _build_batch_request(
            batch,
            dataset_bundle,
            config["serialization_mode"],
            config["model"],
            system_prompt,
        )
        if not first_user_prompt:
            first_user_prompt = request_bundle["user_prompt"]
        request_bundles.append(
            {
                **request_bundle,
                "sequence": batch_index,
                "pairs": batch["pairs"],
                "mode": batch["mode"],
            }
        )

    execution_result = asyncio.run(
        _execute_processing_requests_async(
            run_id,
            config["model"],
            request_bundles,
            blocking_metrics,
            total_pairs,
            config["concurrency"],
        )
    )
    llm_time_s = time.perf_counter() - llm_started
    total_input_tokens = execution_result["total_input_tokens"]
    total_output_tokens = execution_result["total_output_tokens"]
    total_cached_input_tokens = execution_result["total_cached_input_tokens"]
    total_cache_write_tokens = execution_result["total_cache_write_tokens"]
    predicted_pairs = execution_result["predicted_pairs"]
    request_durations = execution_result["request_durations"]
    first_response_text = execution_result["first_response_text"]
    decision_scores = execution_result["decision_scores"]

    metrics = evaluate(
        predicted_pairs,
        dataset_bundle["ground_truth"],
        dataset_bundle["n_records_a"],
        dataset_bundle["n_records_b"],
        labeled_pairs=dataset_bundle.get("labeled_pairs"),
    )
    true_positive_pairs = predicted_pairs & dataset_bundle["ground_truth"]
    false_positive_pairs = predicted_pairs - dataset_bundle["ground_truth"]
    false_negative_pairs = dataset_bundle["ground_truth"] - predicted_pairs
    recall_loss = round(blocking_metrics["recall"] - metrics["recall"], 6)

    cost = estimate_usage_cost(
        config["model"],
        total_input_tokens,
        total_output_tokens,
        cached_input_tokens=total_cached_input_tokens,
        request_count=total_requests,
    )

    total_time_s = time.perf_counter() - started_at
    completed_at = _utcnow_iso()

    result = {
        "run_id": run_id,
        "status": "completed",
        "is_binned": False,
        "created_at": created_at,
        "completed_at": completed_at,
        "dataset": config["dataset"],
        "dataset_entity_type": dataset_bundle["entity_type"],
        "repr_label": candidate_result["repr_label"],
        "blocking_config": {
            "representation": config["representation"],
            "repr_label": candidate_result["repr_label"],
            "index_type": config["index_type"],
            "retrieval": config["retrieval"],
            "k": config["k"],
            "threshold": config["threshold"],
        },
        "processing_config": {
            "serialization_mode": config["serialization_mode"],
            "base_prompt": config["base_prompt"],
            "custom_instructions": config["custom_instructions"],
            "model": config["model"],
            "batching_mode": config["batching_mode"],
            "batch_size": config["batch_size"],
            "concurrency": config["concurrency"],
            "decision_threshold": PROCESSING_MODELS[config["model"]].get("decision_threshold"),
        },
        "counts": {
            "n_records_a": dataset_bundle["n_records_a"],
            "n_records_b": dataset_bundle["n_records_b"],
            "n_ground_truth": len(dataset_bundle["ground_truth"]),
            "candidate_pairs": len(candidate_pairs),
            "avg_candidates_per_record": blocking_metrics["avg_k"],
            "predicted_matches": len(predicted_pairs),
            "llm_requests": total_requests,
            "true_positives": len(true_positive_pairs),
            "false_positives": len(false_positive_pairs),
            "false_negatives": len(false_negative_pairs),
        },
        "blocking_metrics": blocking_metrics,
        "metrics": metrics,
        "recall_loss": recall_loss,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cached_input_tokens": total_cached_input_tokens,
            "cache_write_tokens": total_cache_write_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_input_tokens_per_request": round(total_input_tokens / total_requests, 2) if total_requests else 0.0,
            "avg_output_tokens_per_request": round(total_output_tokens / total_requests, 2) if total_requests else 0.0,
            "avg_tokens_per_pair": round((total_input_tokens + total_output_tokens) / len(candidate_pairs), 2) if candidate_pairs else 0.0,
        },
        "cost": cost,
        "timing": {
            "repr_time_s": candidate_result["repr_time_s"],
            "index_build_time_s": candidate_result["index_build_time_s"],
            "query_time_s": candidate_result["query_time_s"],
            "llm_time_s": round(llm_time_s, 3),
            "avg_request_time_s": round(sum(request_durations) / len(request_durations), 3) if request_durations else 0.0,
            "total_time_s": round(total_time_s, 3),
        },
        "prompt_preview": {
            "system_prompt": system_prompt,
            "first_user_prompt": first_user_prompt,
            "first_response_text": first_response_text,
        },
        "decision_breakdown": {
            "true_positives": _build_pair_debug_rows(true_positive_pairs, dataset_bundle),
            "false_positives": _build_pair_debug_rows(false_positive_pairs, dataset_bundle),
            "false_negatives": _build_pair_debug_rows(false_negative_pairs, dataset_bundle),
        },
        "decision_scores": [
            {
                "record_a_id": pair[0],
                "record_b_id": pair[1],
                "score": score,
            }
            for pair, score in sorted(decision_scores.items())
        ],
        "progress": {
            "stage": "complete",
            "completed_requests": total_requests,
            "total_requests": total_requests,
            "completed_pairs": total_pairs,
            "total_pairs": total_pairs,
            "candidate_pairs": total_pairs,
            "candidate_recall": blocking_metrics["recall"],
        },
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "estimated_cost_usd": cost["total_cost_usd"],
        "total_time_s": round(total_time_s, 3),
    }
    return result
