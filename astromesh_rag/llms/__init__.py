"""LLMs. Backends import their extra lazily, in __init__."""

from .openai_compat import OpenAICompatLLM

__all__ = ["OpenAICompatLLM"]
