from .hybrid import hybrid
from .hyde import HYDE_PROMPT, HypotheticalDocument, hyde
from .multi_query import MULTI_QUERY_PROMPT, LLMQueryVariants, multi_query
from .naive import naive

PATTERNS = {"naive": naive, "hybrid": hybrid, "multi_query": multi_query, "hyde": hyde}

__all__ = [
    "HYDE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "PATTERNS",
    "HypotheticalDocument",
    "LLMQueryVariants",
    "hybrid",
    "hyde",
    "multi_query",
    "naive",
]
