"""The judge contract: what an adapter must provide, and what the router does when it does not."""

from collections.abc import Mapping

import pytest

from langchain_loadout import Answer, JudgeMisconfigured, JudgeUnavailable, Pick, Skill, SkillRouter, Turn, YesNo
from langchain_loadout.testing import ScriptedJudge, check_judge
from langchain_loadout.testing.conformance import SKILLS


def skill(name: str) -> Skill:
    async def read() -> str:
        return f"# {name}\nInstructions for {name}."

    return Skill(name=name, description=f"Description of {name}", read=read)


CATALOG = [skill(n) for n in ("visa-statement", "spending-by-category")]


def answering(**scores: float) -> ScriptedJudge:
    """A judge that scores every option and every yes/no question from `scores`, defaulting to zero."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                out[key] = Answer({o: scores.get(o.replace("-", "_"), 0.0) for o in q.options})
            else:
                out[key] = Answer({"yes": scores.get(key.replace("-", "_"), 0.9)})
        return out

    return ScriptedJudge(answer)


# --- what the router requires -----------------------------------------------------------------------


def test_a_judge_without_limits_is_rejected_when_the_router_is_built():
    class NoLimits:
        async def ask(self, state, questions):
            return {}

    with pytest.raises(JudgeMisconfigured, match="limits"):
        SkillRouter(CATALOG, NoLimits())


async def test_a_misconfigured_judge_surfaces_instead_of_falling_back():
    def refuse(state, questions):
        raise JudgeMisconfigured("the key was rejected")

    with pytest.raises(JudgeMisconfigured, match="rejected"):
        await SkillRouter(CATALOG, ScriptedJudge(refuse)).decide(Turn("a statement"))


async def test_an_unavailable_judge_falls_back_and_changes_nothing():
    def unavailable(state, questions):
        raise JudgeUnavailable("the provider timed out")

    decision = await SkillRouter(CATALOG, ScriptedJudge(unavailable)).decide(Turn("a statement"))
    assert (decision.load, decision.suggest) == ((), ())
    assert "the provider timed out" in decision.trace.failure


async def test_search_also_surfaces_a_misconfigured_judge():
    def refuse(state, questions):
        raise JudgeMisconfigured("the key was rejected")

    with pytest.raises(JudgeMisconfigured):
        await SkillRouter(CATALOG, ScriptedJudge(refuse)).search("a statement")


# --- the conformance kit ----------------------------------------------------------------------------


async def test_an_adapter_that_meets_the_contract_passes():
    await check_judge(answering(spending_by_category=0.9, card_limits=0.1, visa_statement=0.05, about_transfer=0.02))


async def test_missing_limits_is_reported():
    judge = answering(spending_by_category=0.9)
    del judge.limits
    with pytest.raises(AssertionError, match="limits"):
        await check_judge(judge)


async def test_answering_only_the_winning_option_is_reported():
    def argmax_only(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        return {
            key: Answer({"spending-by-category": 1.0}) if isinstance(q, Pick) else Answer({"yes": 0.9}) for key, q in questions.items()
        }

    with pytest.raises(AssertionError, match="probability for every option"):
        await check_judge(ScriptedJudge(argmax_only))


async def test_a_probability_outside_the_range_is_reported():
    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        await check_judge(answering(spending_by_category=1.4, card_limits=0.1, visa_statement=0.05))


async def test_leaving_a_question_unanswered_is_reported():
    def partial(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        return {"pick": Answer(dict.fromkeys(SKILLS, 0.5))}

    with pytest.raises(AssertionError, match="one call answered"):
        await check_judge(ScriptedJudge(partial))


async def test_uncalibrated_answers_are_reported():
    judge = answering(spending_by_category=0.9, card_limits=0.1, visa_statement=0.05, about_spending=0.1)
    with pytest.raises(AssertionError, match="calibrated"):
        await check_judge(judge)
