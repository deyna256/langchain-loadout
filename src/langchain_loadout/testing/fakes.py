"""A judge that answers from a script instead of calling a provider."""

from collections.abc import Callable, Mapping

from langchain_loadout.core.judge import Answer, Limits, Pick, YesNo

Questions = Mapping[str, Pick | YesNo]
Script = Callable[[Mapping[str, object], Questions], Mapping[str, Answer]]


class ScriptedJudge:
    """Answers whatever `script` returns and remembers every call, so a test can assert on them."""

    def __init__(self, script: Script, limits: Limits | None = None) -> None:
        self.script = script
        self.limits = limits or Limits()
        self.calls: list[tuple[Mapping[str, object], Questions]] = []

    async def ask(self, state: Mapping[str, object], questions: Questions) -> Mapping[str, Answer]:
        self.calls.append((state, questions))
        return self.script(state, questions)


def yes(p: float) -> Answer:
    """The answer to a `YesNo` question, with `p` as the probability of yes."""
    return Answer({"yes": p})
