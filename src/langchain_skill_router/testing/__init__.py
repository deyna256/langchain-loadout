"""Tools for people writing a judge adapter or testing an agent that uses Skill Router.

`check_judge` runs an adapter against the `Judge` contract, and `ScriptedJudge` answers whatever a test
tells it to, so an agent can be exercised without calling a provider at all.
"""

from langchain_skill_router.testing.conformance import check_judge
from langchain_skill_router.testing.fakes import ScriptedJudge, yes

__all__ = ["ScriptedJudge", "check_judge", "yes"]
