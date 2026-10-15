import importlib

from .errors import MissingExtraError


def require(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingExtraError(extra, module) from e


def raise_for_status(r) -> None:
    """Like httpx's, but the message carries the (truncated) response body."""
    try:
        r.raise_for_status()
    except Exception as e:
        raise type(e)(f"{e}: {r.text[:500]}", request=e.request, response=r) from e
