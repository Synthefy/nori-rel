"""Compatibility imports; implementations live in ``feature.dfs``."""

from .feature.dfs import CUTOFF_COLUMN as _CUTOFF
from .feature.dfs import FeatureMatrices
from .feature.dfs.fastdfs import FastDFS as RelationalFeatureEngine

__all__ = ["_CUTOFF", "FeatureMatrices", "RelationalFeatureEngine"]
