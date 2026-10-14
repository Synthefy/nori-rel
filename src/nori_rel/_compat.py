"""Narrow pkg_resources compatibility for dependencies that only scan plugins."""

from __future__ import annotations

import sys
import warnings
from importlib.metadata import entry_points
from types import ModuleType


def install_pkg_resources_compatibility() -> None:
    """Supply only iter_entry_points when legacy setuptools is unavailable."""
    try:
        with warnings.catch_warnings():
            # Recent setuptools warns that pkg_resources is deprecated; the
            # import is only a presence check, so the warning is no use here.
            warnings.simplefilter("ignore", UserWarning)
            import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        compatibility = ModuleType("pkg_resources")
        compatibility.iter_entry_points = lambda group: entry_points().select(
            group=group
        )
        sys.modules["pkg_resources"] = compatibility
