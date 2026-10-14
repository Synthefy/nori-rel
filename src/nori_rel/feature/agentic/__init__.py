"""Agent-written SQL feature programs scored by Nori, as a ``FeatureBackend``.

Agents subclass ``ProgramAgentBase`` from ``_agent_base``. ``agent_claude`` and
``agent_openai`` hold one subclass each. Import them from their own modules so
each model provider SDK stays optional.
"""

from ._agent_base import SYSTEM_PROMPT, ProgramAgentBase, ToolSpec
from .backend import AgenticFeatures
from .contracts import (
    AgenticFeatureConfig,
    FeatureProgram,
    FeatureQuery,
    ProgramAgent,
    ProgramScorer,
    ScoreReport,
    TaskKind,
    Trial,
)
from .executor import ProgramExecutor
from .guard import LeakFinding, point_in_time_findings, static_findings
from .scoring import RidgeScorer
from .session import PlanningSession, ProgramRejected

__all__ = [
    "SYSTEM_PROMPT",
    "AgenticFeatureConfig",
    "AgenticFeatures",
    "FeatureProgram",
    "FeatureQuery",
    "LeakFinding",
    "PlanningSession",
    "ProgramAgent",
    "ProgramAgentBase",
    "ProgramExecutor",
    "ProgramRejected",
    "ProgramScorer",
    "RidgeScorer",
    "ScoreReport",
    "TaskKind",
    "ToolSpec",
    "Trial",
    "point_in_time_findings",
    "static_findings",
]
