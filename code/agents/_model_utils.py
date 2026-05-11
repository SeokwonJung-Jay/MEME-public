"""Helpers for ablation-specific model handling.

Main evaluation runs on `gpt-4.1-mini` (OpenAI, legacy chat-completions).
Two ablation branches exist:
  1. OpenAI reasoning models (gpt-5 / o-series) — need max_completion_tokens
     + reasoning_effort; detected by `is_reasoning_model`.
  2. OpenRouter models (e.g., `z-ai/glm-5.1`) — routed to OpenRouter
     endpoint; detected by `is_openrouter_model`.

Callers branch to ablation paths only for these cases; the main code path
is untouched for `gpt-4.1-mini`.
"""

import os

DEFAULT_REASONING_EFFORT = "minimal"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def is_reasoning_model(model: str) -> bool:
    """Return True if `model` is a reasoning model that requires
    max_completion_tokens + reasoning_effort (and rejects temperature).

    Covers OpenAI gpt-5/o-series and OpenRouter GLM-5/GLM-4.7 families.
    """
    if not model:
        return False
    m = model.lower()
    # OpenAI reasoning models
    if any(prefix in m for prefix in ("gpt-5", "o1", "o3-", "o3", "o4-")):
        return True
    # OpenRouter GLM reasoning models (glm-5 / glm-5.1 / glm-5-turbo / glm-4.7-flash)
    if any(tag in m for tag in ("glm-5", "glm-4.7-flash")):
        return True
    return False


def is_openrouter_model(model: str) -> bool:
    """Return True if `model` uses OpenRouter provider-prefix notation
    (e.g., `z-ai/glm-5.1`, `anthropic/claude-3.5-sonnet`).
    Plain OpenAI model names (`gpt-4.1-mini`, `gpt-5`) return False.
    """
    if not model:
        return False
    return "/" in model


def make_openai_client(model: str, api_key: str = None):
    """Create an OpenAI SDK client routed to the correct endpoint for `model`.

    - OpenRouter model → base_url=openrouter, key=OPENROUTER_API_KEY
    - Plain OpenAI model → default (no base_url, key=OPENAI_API_KEY)
    """
    from openai import OpenAI
    if is_openrouter_model(model):
        return OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )
    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
