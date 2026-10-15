from benchmark.serialization import SERIALIZATION_MODES as ALL_SERIALIZATION_MODES


SERIALIZATION_MODES = {
    key: ALL_SERIALIZATION_MODES[key]
    for key in (
        "labels_inline",
        "labels_multiline",
        "json_compact",
        "values_inline",
        "values_multiline",
    )
}


DEFAULT_BASE_PROMPT = """You determine whether two {type_of_data} records represent the same real-world entity.

Account for common variations such as typos, abbreviations, formatting differences, and domain-specific naming conventions.
Make a conservative decision when key identifiers conflict, and keep your reasoning brief and user-facing.
"""


PROCESSING_MODELS = {
    "gpt-5.6-luna": {
        "label": "5.6 Luna",
        "api_model": "gpt-5.6-luna",
        "description": "Efficient GPT-5.6 model for high-volume duplicate classification.",
        "uses_web_search": False,
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 1.20,
        "cached_input_cost_per_million": 0.02,
        "request_cost_usd": 0.0,
    },
    "gpt-5.4-nano": {
        "label": "5.4 Nano",
        "api_model": "gpt-5.4-nano",
        "description": "Cheapest GPT-5.4-class model for simple, high-volume duplicate classification.",
        "uses_web_search": False,
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 1.25,
        "cached_input_cost_per_million": 0.02,
        "request_cost_usd": 0.0,
    },
    "gpt-5.4-mini": {
        "label": "5.4 Mini",
        "api_model": "gpt-5.4-mini",
        "description": "Stronger mini model for coding-style and higher-accuracy duplicate classification.",
        "uses_web_search": False,
        "input_cost_per_million": 0.75,
        "output_cost_per_million": 4.50,
        "cached_input_cost_per_million": 0.075,
        "request_cost_usd": 0.0,
    },
    "gpt-5.4": {
        "label": "5.4",
        "api_model": "gpt-5.4",
        "description": "Flagship GPT-5.4 model for the strongest ER decisions in this benchmark.",
        "uses_web_search": False,
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 15.00,
        "cached_input_cost_per_million": 0.25,
        "request_cost_usd": 0.0,
    },
    "gpt-5.4-websearch": {
        "label": "5.4 Web",
        "api_model": "gpt-5.4",
        "description": "GPT-5.4 with web search enabled for public-entity verification.",
        "uses_web_search": True,
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 15.00,
        "cached_input_cost_per_million": 0.25,
        "request_cost_usd": 0.0,
    },
    "gpt-5-nano": {
        "label": "Nano",
        "api_model": "gpt-5-nano",
        "description": "Fastest and cheapest option for broad benchmark sweeps.",
        "uses_web_search": False,
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.40,
        "cached_input_cost_per_million": 0.005,
        "request_cost_usd": 0.0,
    },
    "gpt-5-mini": {
        "label": "Mini",
        "api_model": "gpt-5-mini",
        "description": "Balanced model for higher-quality duplicate classification.",
        "uses_web_search": False,
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 2.00,
        "cached_input_cost_per_million": 0.025,
        "request_cost_usd": 0.0,
    },
    "gpt-5": {
        "label": "Pro",
        "api_model": "gpt-5",
        "description": "Highest-accuracy benchmark option.",
        "uses_web_search": False,
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 10.00,
        "cached_input_cost_per_million": 0.125,
        "request_cost_usd": 0.0,
    },
    "gpt-5-websearch": {
        "label": "Web",
        "api_model": "gpt-5",
        "description": "Pro model with web search enabled for public-entity verification.",
        "uses_web_search": True,
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 10.00,
        "cached_input_cost_per_million": 0.125,
        "request_cost_usd": 0.0,
    },
    "deepseek-chat": {
        "label": "DeepSeek Chat",
        "api_model": "deepseek-chat",
        "provider": "deepseek",
        "description": "DeepSeek-V3.2 non-thinking mode via the official DeepSeek API.",
        "uses_web_search": False,
        "input_cost_per_million": 0.27,
        "output_cost_per_million": 1.10,
        "cached_input_cost_per_million": 0.07,
        "request_cost_usd": 0.0,
    },
    "deepseek-reasoner": {
        "label": "DeepSeek Reasoner",
        "api_model": "deepseek-reasoner",
        "provider": "deepseek",
        "description": "DeepSeek-V3.2 reasoning mode via the official DeepSeek API.",
        "uses_web_search": False,
        "input_cost_per_million": 0.55,
        "output_cost_per_million": 2.19,
        "cached_input_cost_per_million": 0.14,
        "request_cost_usd": 0.0,
    },
    "grok-4-1-fast-non-reasoning": {
        "label": "Grok 4.1 Fast",
        "api_model": "grok-4-1-fast-non-reasoning",
        "provider": "xai",
        "description": "xAI Grok 4.1 Fast non-reasoning mode via the direct xAI API.",
        "uses_web_search": False,
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.50,
        "cached_input_cost_per_million": 0.0,
        "request_cost_usd": 0.0,
    },
    "grok-4-1-fast-reasoning": {
        "label": "Grok 4.1 Fast Reasoning",
        "api_model": "grok-4-1-fast-reasoning",
        "provider": "xai",
        "description": "xAI Grok 4.1 Fast reasoning mode via the direct xAI API.",
        "uses_web_search": False,
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.50,
        "cached_input_cost_per_million": 0.0,
        "request_cost_usd": 0.0,
    },
    "gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro Preview",
        "api_model": "gemini-3.1-pro-preview",
        "provider": "google",
        "description": "Google Gemini 3.1 Pro Preview via the direct Gemini API.",
        "uses_web_search": False,
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cached_input_cost_per_million": 0.0,
        "request_cost_usd": 0.0,
    },
    "gemini-3-flash-preview": {
        "label": "Gemini 3 Flash Preview",
        "api_model": "gemini-3-flash-preview",
        "provider": "google",
        "description": "Google Gemini 3 Flash Preview via the direct Gemini API.",
        "uses_web_search": False,
        "input_cost_per_million": 0.50,
        "output_cost_per_million": 3.00,
        "cached_input_cost_per_million": 0.0,
        "request_cost_usd": 0.0,
    },
    "jev-1.13.0": {
        "label": "Jev 1.13",
        "api_model": "jev-1.13.0",
        "provider": "typesafe",
        "description": "TypeSafe Jev 1.13 using native System One Noul decisions.",
        "uses_web_search": False,
        "input_cost_per_million": 0.042,
        "output_cost_per_million": 0.0,
        "cached_input_cost_per_million": 0.0,
        "request_cost_usd": 0.0,
        "decision_threshold": 0.5,
    },
}

BATCHING_MODES = {
    "pairwise": "Pairwise",
    "multi_pair": "Multi-pair",
    "one_to_many": "One-to-many",
}


def get_model_config(name):
    if name not in PROCESSING_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return PROCESSING_MODELS[name]


def get_api_model(name):
    return get_model_config(name)["api_model"]


def uses_web_search(name):
    return bool(get_model_config(name)["uses_web_search"])


def get_model_provider(name):
    return get_model_config(name).get("provider", "openai")


def estimate_usage_cost(model_name, input_tokens, output_tokens, cached_input_tokens=0, request_count=0):
    config = get_model_config(model_name)
    input_cost = (max(input_tokens - cached_input_tokens, 0) / 1_000_000) * config["input_cost_per_million"]
    cached_cost = (cached_input_tokens / 1_000_000) * config["cached_input_cost_per_million"]
    output_cost = (output_tokens / 1_000_000) * config["output_cost_per_million"]
    request_cost = request_count * config.get("request_cost_usd", 0.0)
    total = input_cost + cached_cost + output_cost + request_cost
    return {
        "input_cost_usd": round(input_cost, 6),
        "cached_input_cost_usd": round(cached_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "request_cost_usd": round(request_cost, 6),
        "total_cost_usd": round(total, 6),
    }
