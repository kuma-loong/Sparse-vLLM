from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparsevllm.llm import LLM
    from sparsevllm.sampling_params import SamplingParams


def __getattr__(name: str):
    if name == "LLM":
        from sparsevllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from sparsevllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLM", "SamplingParams"]
