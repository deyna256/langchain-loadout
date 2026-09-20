"""Per-turn skill selection for LangChain and deepagents agents.

Everything listed in `__all__` is the public API; anything else may change without notice.

The root exports the core only, so importing the package pulls in no framework and no provider SDK.
The parts that do are imported from their own modules:

- `langchain_loadout.langchain.middleware` — `LoadoutSkillsMiddleware`, which replaces
  `SkillsMiddleware` in a deepagents agent;
- `langchain_loadout.providers.jev` — `JevJudge`, the adapter for Jev (TypeSafe System One), which
  needs the `jev` extra and `TYPESAFE_API_KEY`.

Start from the README; the reasoning behind the design is in docs/design.md.
"""

from langchain_loadout.core.judge import (
    Answer,
    Judge,
    JudgeMisconfigured,
    JudgeUnavailable,
    Limits,
    LoadoutError,
    Pick,
    YesNo,
)
from langchain_loadout.core.router import SkillRouter, decide_from_trace
from langchain_loadout.core.types import DEFAULTS, Decision, Settings, Skill, Trace, Turn

__all__ = [
    "DEFAULTS",
    "Answer",
    "Decision",
    "Judge",
    "JudgeMisconfigured",
    "JudgeUnavailable",
    "Limits",
    "LoadoutError",
    "Pick",
    "Settings",
    "Skill",
    "SkillRouter",
    "Trace",
    "Turn",
    "YesNo",
    "decide_from_trace",
]
