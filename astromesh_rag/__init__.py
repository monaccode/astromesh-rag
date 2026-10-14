"""astromesh-rag: composable RAG pipelines."""

from .errors import MissingExtraError, RAGError, SpecError, StageError
from .registry import BuildContext, Registry, registry
from .types import Chunk, Document, Hit, QueryResult, chunk_id

__version__ = "0.1.0"

__all__ = [
    "BuildContext",
    "Chunk",
    "Document",
    "Hit",
    "MissingExtraError",
    "QueryResult",
    "RAGError",
    "Registry",
    "SpecError",
    "StageError",
    "__version__",
    "chunk_id",
    "registry",
]
