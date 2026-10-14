"""Credential-free database execution and resource-limit errors."""


class ConnectorError(RuntimeError):
    """Database connection, discovery, or extraction failed."""


class SnapshotLimitError(ConnectorError):
    """A database read exceeded a configured resource ceiling."""
