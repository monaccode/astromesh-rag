class RAGError(Exception):
    """Base class for every astromesh-rag error."""


class SpecError(RAGError):
    """Invalid or unknown pipeline spec."""


class MissingExtraError(RAGError):
    def __init__(self, extra: str, module: str):
        super().__init__(f"{module} is not installed: pip install 'astromesh-rag[{extra}]'")
        self.extra = extra
        self.module = module


class StageError(RAGError):
    def __init__(self, stage: str, name: str, cause: Exception):
        super().__init__(f"{stage} '{name}' failed: {str(cause) or type(cause).__name__}")
        self.stage = stage
        self.name = name
        self.cause = cause
