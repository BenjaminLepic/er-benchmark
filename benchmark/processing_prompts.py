import json
from pydantic import BaseModel

from benchmark.processing_models import (
    DEFAULT_BASE_PROMPT,
    get_api_model,
    get_model_provider,
    uses_web_search,
)
from benchmark.serialization import serialize_record

WEB_SEARCH_INCLUDE = ["web_search_call.action.sources"]
WEB_SEARCH_GUIDANCE = """
Web verification guidance:
- When the records describe public people, companies, organizations, addresses, websites, or other public entities, use web search to verify relevant facts before deciding.
- If web results are inconclusive, fall back to the record content and do not overfit on weak public matches.
""".strip()


PAIRWISE_SCHEMA = {
    "type": "json_schema",
    "name": "pairwise_duplicate_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_duplicate": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_duplicate", "reason"],
        "additionalProperties": False,
    },
}

MULTI_PAIR_SCHEMA = {
    "type": "json_schema",
    "name": "multi_pair_duplicate_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_id": {"type": "string"},
                        "is_duplicate": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["pair_id", "is_duplicate", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
}

ONE_TO_MANY_SCHEMA = {
    "type": "json_schema",
    "name": "one_to_many_duplicate_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "is_duplicate": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "is_duplicate", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
}


class PairwiseDecision(BaseModel):
    is_duplicate: bool
    reason: str


class MultiPairDecisionItem(BaseModel):
    pair_id: str
    is_duplicate: bool
    reason: str


class MultiPairDecision(BaseModel):
    decisions: list[MultiPairDecisionItem]


class OneToManyDecisionItem(BaseModel):
    candidate_id: str
    is_duplicate: bool
    reason: str


class OneToManyDecision(BaseModel):
    decisions: list[OneToManyDecisionItem]


def schema_to_pydantic_model(schema):
    schema_name = schema.get("name")
    if schema_name == "pairwise_duplicate_decision":
        return PairwiseDecision
    if schema_name == "multi_pair_duplicate_decision":
        return MultiPairDecision
    if schema_name == "one_to_many_duplicate_decision":
        return OneToManyDecision
    raise ValueError(f"Unsupported schema for structured parsing: {schema_name}")
def build_system_prompt(base_prompt, type_of_data, custom_instructions=""):
    prompt = (base_prompt or DEFAULT_BASE_PROMPT).replace("{type_of_data}", type_of_data)
    prompt = prompt.strip()
    if custom_instructions and custom_instructions.strip():
        prompt += (
            "\n\nAdditional user instructions (follow these when making your evaluation):\n"
            f"{custom_instructions.strip()}"
        )
    prompt += "\n\nReturn only the structured decision output for the records you evaluate."
    return prompt


def build_pairwise_prompt(serialized_a, serialized_b):
    return (
        "Compare these potential duplicate records.\n\n"
        f"Record A:\n{serialized_a}\n\n"
        f"Record B:\n{serialized_b}\n"
    )


def build_multi_pair_prompt(pairs):
    lines = [
        "Evaluate each candidate pair independently.",
        "Return one decision for every pair_id.",
        "",
    ]
    for pair in pairs:
        lines.extend(
            [
                f"Pair ID: {pair['pair_id']}",
                f"Record A:\n{pair['record_a']}",
                f"Record B:\n{pair['record_b']}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_one_to_many_prompt(serialized_a, candidates):
    lines = [
        "Evaluate each candidate independently against the shared source record.",
        "Return one decision for every candidate_id.",
        "",
        f"Source Record:\n{serialized_a}",
        "",
        "Candidates:",
        "",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"Candidate ID: {candidate['candidate_id']}",
                f"{candidate['record_b']}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _schema_example(schema):
    schema_name = schema.get("name")
    if schema_name == "pairwise_duplicate_decision":
        return '{"is_duplicate": false, "reason": "brief explanation"}'
    if schema_name == "multi_pair_duplicate_decision":
        return '{"decisions": [{"pair_id": "pair_0", "is_duplicate": false, "reason": "brief explanation"}]}'
    if schema_name == "one_to_many_duplicate_decision":
        return '{"decisions": [{"candidate_id": "123", "is_duplicate": false, "reason": "brief explanation"}]}'
    return '{"result": "json"}'


def extract_response_text(response):
    response_text = getattr(response, "output_text", None)
    if response_text:
        return response_text

    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text:
        return direct_text.strip()

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump_json"):
                return parsed.model_dump_json()
            if hasattr(parsed, "model_dump"):
                return json.dumps(parsed.model_dump())
            return json.dumps(parsed)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts).strip()

    parts = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            if getattr(content_item, "type", None) == "text" and getattr(content_item, "text", None):
                parts.append(content_item.text)
    return "\n".join(parts).strip()


def extract_usage(response):
    usage = getattr(response, "usage", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage is None and usage_metadata is not None:
        usage = usage_metadata
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
        }

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = (
        getattr(prompt_tokens_details, "cached_tokens", None)
        if prompt_tokens_details is not None
        else None
    )
    prompt_cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
    cache_write_tokens = (
        getattr(prompt_tokens_details, "cache_write_tokens", None)
        if prompt_tokens_details is not None
        else None
    )

    input_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_token_count", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "candidates_token_count", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    cached_input_tokens = (
        prompt_cache_hit_tokens
        or cached_tokens
        or getattr(usage, "cached_content_token_count", None)
        or getattr(usage, "prompt_cache_hit_tokens", None)
        or getattr(usage, "cached_input_tokens", None)
        or getattr(usage, "cached_tokens", None)
        or 0
    )
    cache_write_tokens = (
        cache_write_tokens
        or getattr(usage, "cache_write_tokens", None)
        or 0
    )
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "cache_write_tokens": int(cache_write_tokens),
    }


def parse_json_response(response):
    text = extract_response_text(response)
    if not text:
        raise ValueError("Empty model response")
    return json.loads(text), text


def build_request(model_name, system_prompt, user_prompt, schema):
    provider = get_model_provider(model_name)

    if provider == "typesafe":
        raise ValueError("TypeSafe requests must be built from structured benchmark state")

    if provider == "xai":
        return {
            "request_type": "chat_completions_parse",
            "request_kwargs": {
                "model": get_api_model(model_name),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": schema_to_pydantic_model(schema),
            },
        }

    if provider == "google":
        return {
            "request_type": "google_generate_content",
            "request_kwargs": {
                "model": get_api_model(model_name),
                "contents": user_prompt,
                "config": {
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema["schema"],
                },
            },
        }

    if provider == "deepseek":
        json_instruction = (
            "Return valid JSON only. "
            f"Use this exact shape: {_schema_example(schema)}"
        )
        request_kwargs = {
            "model": get_api_model(model_name),
            "messages": [
                {"role": "system", "content": f"{system_prompt}\n\n{json_instruction}"},
                {"role": "user", "content": user_prompt},
            ],
        }
        request_kwargs["response_format"] = {"type": "json_object"}
        return {
            "request_type": "chat_completions",
            "request_kwargs": request_kwargs,
        }

    request_kwargs = {
        "model": get_api_model(model_name),
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": "low"},
        "text": {"format": schema},
    }

    if uses_web_search(model_name):
        request_kwargs["input"][0]["content"] = f"{system_prompt}\n\n{WEB_SEARCH_GUIDANCE}"
        request_kwargs["tools"] = [{"type": "web_search"}]
        request_kwargs["include"] = WEB_SEARCH_INCLUDE

    return {
        "request_type": "responses",
        "request_kwargs": request_kwargs,
    }
