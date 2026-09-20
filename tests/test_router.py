"""What the router decides on a turn, given the answers the judge is scripted to give."""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from langchain_loadout import Answer, Pick, Settings, Skill, SkillRouter, Trace, Turn, YesNo, decide_from_trace
from langchain_loadout.testing import ScriptedJudge, yes


def skill(name: str, text: str = "") -> Skill:
    async def read() -> str:
        return text or f"# {name}\nInstructions for {name}."

    return Skill(name=name, description=f"Description of {name}", read=read)


CATALOG = [skill(n) for n in ("visa-statement", "spending-by-category", "subscriptions", "card-limits", "dispute")]


def scripted(
    pick: Mapping[str, float],
    need: float = 0.9,
    fits: Mapping[str, float] | None = None,
    still: Mapping[str, float] | None = None,
    beyond: float = 0.9,
) -> ScriptedJudge:
    """A scripted judge: ranking probabilities, plus the need, beyond, fits and still-needed answers."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                out[key] = Answer({o: pick.get(o, 0.0) for o in q.options})
            elif key == "need":
                out[key] = yes(need)
            elif key == "beyond":
                out[key] = yes(beyond)
            elif key.startswith("fits:"):
                out[key] = yes((fits or {}).get(key.removeprefix("fits:"), 0.0))
            elif key.startswith("still:"):
                out[key] = yes((still or {}).get(key.removeprefix("still:"), 0.0))
        return out

    return ScriptedJudge(answer)


async def test_no_skill_needed_loads_nothing_and_keeps_what_is_loaded():
    judge = scripted({"visa-statement": 0.5}, need=0.1, fits={"visa-statement": 0.95})
    d = await SkillRouter(CATALOG, judge).decide(Turn("hello", loaded=("subscriptions",)))
    assert (d.load, d.suggest, d.keep, d.drop) == ((), (), ("subscriptions",), ())


async def test_confident_fit_is_loaded_doubtful_is_suggested():
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits={"visa-statement": 0.92, "spending-by-category": 0.55})
    d = await SkillRouter(CATALOG, judge).decide(Turn("I need a statement for the embassy"))
    assert d.load == ("visa-statement",)
    assert d.suggest == ("spending-by-category",)


async def test_no_more_than_max_load_the_rest_become_suggestions():
    fits = {"visa-statement": 0.95, "spending-by-category": 0.9, "subscriptions": 0.85}
    judge = scripted(dict.fromkeys(fits, 0.3), fits=fits)
    d = await SkillRouter(CATALOG, judge, Settings(max_load=2)).decide(Turn("all of it at once"))
    assert d.load == ("visa-statement", "spending-by-category")
    assert d.suggest == ("subscriptions",)


async def test_loaded_skill_is_dropped_when_no_longer_needed_and_not_reloaded():
    judge = scripted(
        {"spending-by-category": 0.8, "subscriptions": 0.1},
        fits={"spending-by-category": 0.95, "subscriptions": 0.9},
        still={"spending-by-category": 0.9, "subscriptions": 0.1},
    )
    turn = Turn("and for April?", context="spending by category in May", loaded=("spending-by-category", "subscriptions"))
    d = await SkillRouter(CATALOG, judge).decide(turn)
    assert d.keep == ("spending-by-category",)
    assert d.drop == ("subscriptions",)
    assert d.load == ()


async def test_ranking_sees_the_whole_catalog_and_verification_only_the_candidates():
    judge = scripted(
        {"visa-statement": 0.5, "spending-by-category": 0.3, "subscriptions": 0.1, "card-limits": 0.05}, still={"dispute": 0.9}
    )
    await SkillRouter(CATALOG, judge, replace(Settings(), max_candidates=2)).decide(Turn("something", loaded=("dispute",)))
    gate, select, verify = judge.calls
    assert {k for k in gate[1] if k.startswith("still:")} == {"still:dispute"}
    assert set(select[1]["skill"].options) == {s.name for s in CATALOG}
    fits = {k.removeprefix("fits:") for k in verify[1] if k.startswith("fits:")}
    assert fits == {"visa-statement", "spending-by-category"}
    assert not any(k.startswith(("still:", "need")) for k in verify[1])  # these were already asked in the cheap call


async def test_judge_gets_request_context_and_loaded_as_given():
    judge = scripted({"visa-statement": 0.9})
    turn = Turn("and for April?", context="context from the product", loaded=("subscriptions",))
    await SkillRouter(CATALOG, judge).decide(turn)
    state = judge.calls[0][0]
    assert (state["request"], state["context"], state["loaded"]) == ("and for April?", "context from the product", ["subscriptions"])


async def test_verification_sees_the_head_of_the_skill_text():
    long = "x" * 5000
    catalog = [skill("visa-statement", long)]
    judge = scripted({"visa-statement": 1.0}, fits={"visa-statement": 0.9})
    await SkillRouter(catalog, judge, Settings(head_chars=100)).decide(Turn("visa"))
    question = judge.calls[2][1]["fits:visa-statement"].instructions
    assert "x" * 100 in question and "x" * 101 not in question
    assert "candidates" not in judge.calls[2][0]  # a candidate's text lives only in its own question


# --- failures do not break the turn ---------------------------------------------------------------


class Boom(ScriptedJudge):
    """A judge that fails from the given call onwards: 1 is the cheap call, 2 ranking, 3 verification."""

    def __init__(self, fail_on: int, pick: Mapping[str, float], slow: float = 0.0) -> None:
        super().__init__(scripted(pick).script)
        self.fail_on, self.slow = fail_on, slow

    async def ask(self, state, questions):
        if len(self.calls) + 1 >= self.fail_on:
            self.calls.append((state, questions))
            if self.slow:
                import asyncio

                await asyncio.sleep(self.slow)
            raise RuntimeError("provider unavailable")
        return await super().ask(state, questions)


async def test_selection_failure_changes_nothing():
    judge = Boom(fail_on=1, pick={})
    d = await SkillRouter(CATALOG, judge).decide(Turn("something", loaded=("subscriptions",)))
    assert (d.load, d.suggest, d.keep, d.drop) == ((), (), ("subscriptions",), ())
    assert "provider unavailable" in d.trace.failure


async def test_slow_judge_is_cut_by_timeout_and_changes_nothing():
    judge = Boom(fail_on=1, pick={}, slow=5)
    d = await SkillRouter(CATALOG, judge, Settings(timeout=0.05)).decide(Turn("something"))
    assert (d.load, d.suggest) == ((), ())
    assert d.trace.failure == "timeout"


async def test_verification_failure_suggests_best_of_selection():
    judge = Boom(fail_on=3, pick={"visa-statement": 0.6, "spending-by-category": 0.3, "subscriptions": 0.1})
    d = await SkillRouter(CATALOG, judge).decide(Turn("statement"))
    assert d.load == ()
    assert d.suggest == ("visa-statement", "spending-by-category", "subscriptions")


async def test_unreadable_skill_drops_out_of_candidates():
    async def broken() -> str:
        raise OSError("file is gone")

    catalog = [Skill("visa-statement", "Description", broken), *CATALOG[1:]]
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits={"spending-by-category": 0.9})
    d = await SkillRouter(catalog, judge).decide(Turn("statement"))
    assert d.load == ("spending-by-category",)
    assert not any(k == "fits:visa-statement" for k in judge.calls[2][1])


@pytest.mark.parametrize(
    "catalog",
    [
        [skill("dup"), skill("dup")],
        [Skill("empty", "  ", skill("x").read)],
    ],
)
def test_broken_catalog_fails_at_once(catalog):
    with pytest.raises(ValueError):
        SkillRouter(catalog, scripted({}))


# --- search behind find_skill ---------------------------------------------------------------------


async def test_search_returns_best_skills_by_probability():
    judge = scripted({"subscriptions": 0.5, "dispute": 0.3, "card-limits": 0.1, "visa-statement": 0.05})
    found = await SkillRouter(CATALOG, judge).search("charged for a subscription I cancelled", limit=2)
    assert [s.name for s in found] == ["subscriptions", "dispute"]
    assert judge.calls[0][0]["request"] == "charged for a subscription I cancelled"


async def test_search_failure_returns_nothing():
    found = await SkillRouter(CATALOG, Boom(fail_on=1, pick={})).search("something")
    assert found == []


# --- cheap questions first, ranking probabilities in the trace, skipping verification ---------------


async def test_no_skill_needed_is_decided_by_one_cheap_call_without_the_catalog():
    judge = scripted({"visa-statement": 0.9}, need=0.02)
    d = await SkillRouter(CATALOG, judge).decide(Turn("thanks"))
    assert len(judge.calls) == 1 and not any(isinstance(q, Pick) for q in judge.calls[0][1].values())
    assert (d.load, d.suggest, d.trace.stage) == ((), (), "gate")


async def test_continuing_topic_keeps_what_is_needed_without_selection():
    judge = scripted({"visa-statement": 0.9}, beyond=0.1, still={"spending-by-category": 0.9, "subscriptions": 0.1})
    turn = Turn("and for April?", loaded=("spending-by-category", "subscriptions"))
    d = await SkillRouter(CATALOG, judge).decide(turn)
    assert len(judge.calls) == 1
    assert (d.keep, d.drop, d.load) == (("spending-by-category",), ("subscriptions",), ())


async def test_something_beyond_the_loaded_goes_to_selection_and_keeps_what_is_still_needed():
    judge = scripted({"subscriptions": 0.8}, beyond=0.9, still={"spending-by-category": 0.9}, fits={"subscriptions": 0.95})
    d = await SkillRouter(CATALOG, judge).decide(Turn("what about subscriptions?", loaded=("spending-by-category",)))
    assert len(judge.calls) == 3
    assert (d.keep, d.load) == (("spending-by-category",), ("subscriptions",))


async def test_nothing_loaded_asks_no_beyond_question():
    judge = scripted({"visa-statement": 0.9}, fits={"visa-statement": 0.9})
    await SkillRouter(CATALOG, judge).decide(Turn("a statement for a visa"))
    assert "beyond" not in judge.calls[0][1]


async def test_selection_probabilities_are_in_the_trace():
    judge = scripted({"visa-statement": 0.7, "dispute": 0.2}, fits={"visa-statement": 0.9})
    d = await SkillRouter(CATALOG, judge, Settings(max_candidates=2)).decide(Turn("visa"))
    assert d.trace.candidates == (("visa-statement", 0.7), ("dispute", 0.2))
    assert d.trace.stage == "verify"


async def test_confident_selection_skips_verification_only_when_the_product_set_a_threshold():
    judge = scripted({"visa-statement": 0.9, "dispute": 0.05})
    d = await SkillRouter(CATALOG, judge, Settings(skip_verify_at=0.8)).decide(Turn("visa"))
    assert len(judge.calls) == 2 and d.load == ("visa-statement",) and d.trace.stage == "skip"

    unsure = scripted({"visa-statement": 0.6, "dispute": 0.3}, fits={"visa-statement": 0.9})
    await SkillRouter(CATALOG, unsure, Settings(skip_verify_at=0.8)).decide(Turn("visa"))
    assert len(unsure.calls) == 3  # ranking is not confident, so verification runs

    off = scripted({"visa-statement": 0.99}, fits={"visa-statement": 0.9})
    await SkillRouter(CATALOG, off).decide(Turn("visa"))
    assert len(off.calls) == 3  # no threshold set, so verification always runs


def test_skip_threshold_can_be_tried_on_saved_traces():
    trace = Trace(candidates=(("visa-statement", 0.9), ("dispute", 0.05)), need=0.9, fits={"visa-statement": 0.3})
    assert decide_from_trace(Settings(), (), trace).load == ()  # verification says no
    assert decide_from_trace(Settings(skip_verify_at=0.8), (), trace).load == ("visa-statement",)  # the threshold says yes


async def test_every_question_can_be_written_for_the_product_domain():
    settings = Settings(
        need_question="Does this need a bank skill?",
        rank_question="Which bank skill fits best?",
        fits_question="Does {name} do this? {text}",
        still_question="Is {name} still in play?",
        beyond_question="Anything beyond what is loaded?",
    )
    judge = scripted({"visa-statement": 0.9}, fits={"visa-statement": 0.9}, still={"dispute": 0.9})
    await SkillRouter(CATALOG, judge, settings).decide(Turn("a statement", loaded=("dispute",)))
    gate, _, verify = judge.calls
    assert gate[1]["need"].instructions == "Does this need a bank skill?"
    assert gate[1]["beyond"].instructions == "Anything beyond what is loaded?"
    assert gate[1]["still:dispute"].instructions == "Is dispute still in play?"
    assert judge.calls[1][1]["skill"].instructions == "Which bank skill fits best?"
    assert verify[1]["fits:visa-statement"].instructions.startswith("Does visa-statement do this? # visa-statement")
