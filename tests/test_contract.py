"""The judge contract: what an adapter must provide, and what the router does when it does not."""

from collections.abc import Mapping

import pytest
from conftest import skill

from langchain_loadout import Answer, JudgeMisconfigured, JudgeUnavailable, Pick, SkillRouter, Turn, YesNo
from langchain_loadout.testing import ScriptedJudge, check_judge
from langchain_loadout.testing.conformance import SKILLS

CATALOG = [skill(n) for n in ("visa-statement", "spending-by-category")]

# What an adapter that meets the contract answers to the conformance questions.
GOOD = {"spending-by-category": 0.9, "card-limits": 0.1, "visa-statement": 0.05, "about-spending": 0.8, "about-transfer": 0.02}


def answering(scores: Mapping[str, float]) -> ScriptedJudge:
    """A judge that answers the conformance questions from `scores`, giving anything absent a zero."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        return {
            key: Answer({option: scores.get(option, 0.0) for option in q.options})
            if isinstance(q, Pick)
            else Answer({"yes": scores.get(key, 0.0)})
            for key, q in questions.items()
        }

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
    await check_judge(answering(GOOD))


async def test_missing_limits_is_reported():
    judge = answering(GOOD)
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
        await check_judge(answering(GOOD | {"spending-by-category": 1.4}))


async def test_leaving_a_question_unanswered_is_reported():
    def partial(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        return {"pick": Answer(dict.fromkeys(SKILLS, 0.5))}

    with pytest.raises(AssertionError, match="one call answered"):
        await check_judge(ScriptedJudge(partial))


async def test_uncalibrated_answers_are_reported():
    judge = answering(GOOD | {"about-spending": 0.01})  # an obvious yes now scores below an obvious no

    with pytest.raises(AssertionError, match="calibrated"):
        await check_judge(judge)
