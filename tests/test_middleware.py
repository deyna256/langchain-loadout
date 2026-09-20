"""The Loadout wrapper around the deepagents skills middleware: what the model sees, and what it sees on failure."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_loadout import Answer, Pick, YesNo
from langchain_loadout.langchain import LoadoutSkillsMiddleware
from langchain_loadout.testing import ScriptedJudge, yes

SKILLS = {
    "visa-statement": "Statement for a visa: money movement, in English, stamped by the bank.",
    "spending-by-category": "The user's spending by category over a period.",
    "card-limits": "Card limits: withdrawals, transfers, purchases.",
}


class RecordingModel(GenericFakeChatModel):
    """A fake model: returns scripted replies and remembers what it was sent."""

    seen: list[list[Any]] = []  # noqa: RUF012 — a pydantic field, so each instance gets its own list

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def backend(tmp_path: Path) -> FilesystemBackend:
    for name, description in SKILLS.items():
        folder = tmp_path / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f'---\nname: {name}\ndescription: "{description}"\n---\n\n# {name}\nInstruction text for {name}.\n'
        )
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def judge_choosing(name: str | None) -> ScriptedJudge:
    """Picks `name` with confidence, or fails when `name` is None."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        if name is None:
            raise RuntimeError("provider unavailable")
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                out[key] = Answer({o: (0.9 if o == name else 0.05) for o in q.options})
            else:
                out[key] = yes(0.95 if key in ("need", f"fits:{name}") else 0.05)
        return out

    return ScriptedJudge(answer)


async def run(backend: FilesystemBackend, judge: ScriptedJudge, replies: list[AIMessage]) -> RecordingModel:
    model = RecordingModel(messages=iter(replies))
    model.seen = []
    middleware = LoadoutSkillsMiddleware(
        backend=backend, sources=["/skills/"], judge=judge, catalog_hint="Statements, spending, card limits."
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})
    return model


def system_text(messages: list[Any]) -> str:
    return "\n".join(m.text for m in messages if isinstance(m, SystemMessage))


async def test_confident_choice_shows_only_that_skill_and_its_text(backend):

    model = await run(backend, judge_choosing("visa-statement"), [AIMessage("done")])
    prompt = system_text(model.seen[0])

    assert "Instruction text for visa-statement" in prompt
    assert "card-limits" not in prompt and "spending-by-category" not in prompt


async def test_failure_falls_back_to_the_usual_full_list(backend):

    model = await run(backend, judge_choosing(None), [AIMessage("done")])
    prompt = system_text(model.seen[0])

    assert all(name in prompt for name in SKILLS)
    assert "Instruction text for" not in prompt


async def test_decision_is_made_once_per_turn(backend):
    judge = judge_choosing("visa-statement")
    ls = AIMessage("", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call-1"}])

    model = await run(backend, judge, [ls, AIMessage("done")])

    assert len(model.seen) == 2  # the model was called twice in the turn
    assert len(judge.calls) == 3  # the judge once per turn: cheap call, ranking, verification
    assert "Instruction text for visa-statement" in system_text(model.seen[1])


async def test_every_decision_is_reported_for_observability(backend):
    seen = []
    model = RecordingModel(messages=iter([AIMessage("done")]))
    middleware = LoadoutSkillsMiddleware(
        backend=backend, sources=["/skills/"], judge=judge_choosing("visa-statement"), on_decision=seen.append
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])

    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})

    assert [d.load for d in seen] == [("visa-statement",)]
