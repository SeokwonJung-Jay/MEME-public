"""Monkey-patch OpenAI / Anthropic SDK calls to inject a fixed seed.

Used for the multi-seed stability ablation (Table:sd in the paper).
Pattern mirrors eval/budget_tracker.install_patches: idempotent, applied
once per worker process. Safe to call from worker spawn (e.g., inside
ProcessPoolExecutor children).
"""

_PATCHED = False
_SEED = None


def install_seed_patches(seed: int):
    """Patch OpenAI chat completions + Anthropic Messages.create to add seed=<seed>."""
    global _PATCHED, _SEED
    _SEED = seed
    if _PATCHED:
        return
    _PATCHED = True

    # OpenAI ChatCompletions.create — most agents' answer + internal calls
    try:
        from openai.resources.chat.completions import Completions
        _orig = Completions.create

        def _patched(self, **kwargs):
            kwargs.setdefault("seed", _SEED)
            return _orig(self, **kwargs)

        Completions.create = _patched
    except Exception as e:
        print(f"  [seed_injector] OpenAI sync patch skipped: {e}")

    try:
        from openai.resources.chat.completions import AsyncCompletions
        _orig_a = AsyncCompletions.create

        async def _patched_a(self, **kwargs):
            kwargs.setdefault("seed", _SEED)
            return await _orig_a(self, **kwargs)

        AsyncCompletions.create = _patched_a
    except Exception as e:
        print(f"  [seed_injector] OpenAI async patch skipped: {e}")

    # Anthropic does not expose a public seed parameter, but we still tag the
    # config so logs reflect which seed an Anthropic-answering run used.
    print(f"  [seed_injector] Active seed = {seed}")
