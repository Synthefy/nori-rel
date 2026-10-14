"""Errors raised when a saved pipeline cannot be loaded or used."""


class PipelineError(ValueError):
    """A saved pipeline cannot be loaded or does not fit this prediction."""


class PipelineFormatError(PipelineError):
    """The file is not a pipeline this nori-rel can read, or it was edited."""


class PipelineTaskError(PipelineError):
    """The request asks for a different task than the pipeline predicts."""


class PipelineSchemaError(PipelineError):
    """The connected database no longer has the schema the pipeline read."""


class PipelineContextError(PipelineError):
    """The context rows are not the ones the request asked to reproduce."""


class PipelineFeatureError(PipelineError):
    """The saved features cannot run, or they produced different columns."""


class PipelineModelError(PipelineError):
    """The client or the local Nori runtime differs from the saved model."""


class PipelineVersionWarning(UserWarning):
    """The pipeline was saved by another version of nori-rel."""


__all__ = [
    "PipelineContextError",
    "PipelineError",
    "PipelineFeatureError",
    "PipelineFormatError",
    "PipelineModelError",
    "PipelineSchemaError",
    "PipelineTaskError",
    "PipelineVersionWarning",
]
