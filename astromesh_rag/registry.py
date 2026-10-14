"""Name -> factory registry for pluggable stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .errors import SpecError
from .stages import LLM, Embedder

KINDS = ("splitter", "embedder", "index", "reranker", "llm")


@dataclass
class BuildContext:
    dimension: int | None = None
    embedder: Embedder | None = None
    llm: LLM | None = None


class Registry:
    def __init__(self) -> None:
        self._f: dict[str, dict[str, Callable[[dict, BuildContext], object]]] = {
            k: {} for k in KINDS
        }

    def _kind(self, kind: str) -> dict:
        if kind not in self._f:
            raise SpecError(f"unknown kind '{kind}'; known: {', '.join(KINDS)}")
        return self._f[kind]

    def register(
        self, kind: str, name: str, factory: Callable[[dict, BuildContext], object]
    ) -> None:
        self._kind(kind)[name] = factory

    def build(self, kind: str, name: str, cfg: dict, ctx: BuildContext) -> object:
        table = self._kind(kind)
        if name not in table:
            raise SpecError(f"unknown {kind} '{name}'; known: {', '.join(sorted(table))}")
        return table[name](cfg, ctx)

    def names(self, kind: str) -> list[str]:
        return sorted(self._kind(kind))


registry = Registry()
