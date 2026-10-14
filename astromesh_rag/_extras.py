import importlib

from .errors import MissingExtraError


def require(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingExtraError(extra, module) from e
