"""Catalog and context larger than the provider allows: ranking is split and every call stays inside."""

from collections.abc import Mapping

import pytest
from conftest import skill

from langchain_loadout import Answer, JudgeMisconfigured, Limits, Pick, Settings, SkillRouter, Turn, YesNo
from langchain_loadout.testing import ScriptedJudge, yes

TEXT = "Instructions."

# three groups of four skills; each option costs about (name + description) characters
CATALOG = [
    skill(f"{g}-{i}", group=g, text=TEXT, description=f"Description of {g}-{i} " * 20)
    for g in ("docs", "spend", "tax")
    for i in range(4)
]
SMALL = Limits(max_tokens=4200, max_options=255, tokens_per_char=1.0)  # the whole catalog cannot fit one call
FIT = Settings(max_candidates=3, head_chars=50, request_chars=200, context_chars=100)  # verification fits SMALL


def judge(best: str, limits: Limits, fail_chunk_with: str | None = None) -> ScriptedJudge:
    """Prefers `best` in every pick, ranks the rest below it, and verifies `best` as a fit."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                if fail_chunk_with in q.options and len(q.options) > 2:
                    raise RuntimeError("request too large")
                names = sorted(q.options, key=lambda n: (n != best, n))
                out[key] = Answer({n: 1 / (i + 2) for i, n in enumerate(names)})
            elif key == "need":
                out[key] = yes(0.9)
            else:
                out[key] = yes(0.95 if key == f"fits:{best}" else 0.05)
        return out

    return ScriptedJudge(answer, limits)


def estimate(fake: ScriptedJudge, state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> float:
    """The same size estimate the router makes: characters times tokens per character."""
    text = str(state) + "".join(
        q.instructions + "".join(f"{k}{v}" for k, v in getattr(q, "options", {}).items()) for q in questions.values()
    )
    return len(text) * fake.limits.tokens_per_char


def picks(fake: ScriptedJudge) -> list[Pick]:
    """The ranking picks: over the catalog, its parts and their merge. Verification's pick is not one of them."""
    return [q for _, qs in fake.calls for key, q in qs.items() if isinstance(q, Pick) and key == "skill"]


def first_round(fake: ScriptedJudge) -> list[Pick]:
    """The first ranking round: the catalog parts, which run before the merge."""
    seen: set[str] = set()
    out = []
    for p in picks(fake):
        if seen >= {s.name for s in CATALOG}:
            break
        out.append(p)
        seen |= set(p.options)
    return out


async def test_catalog_that_fits_is_chosen_in_one_call_as_before():
    fake = judge("spend-2", Limits())

    d = await SkillRouter(CATALOG, fake).decide(Turn("spending"))

    assert len(picks(fake)) == 1 and d.load == ("spend-2",)


async def test_big_catalog_is_split_every_call_fits_and_the_best_still_wins():
    fake = judge("tax-3", SMALL)

    d = await SkillRouter(CATALOG, fake, FIT).decide(Turn("taxes"))

    assert d.load == ("tax-3",)
    assert len(picks(fake)) > 2  # the parts plus the merge
    for state, questions in fake.calls:
        assert estimate(fake, state, questions) <= SMALL.max_tokens
    all_options = [n for p in first_round(fake) for n in p.options]
    assert sorted(all_options) == sorted(s.name for s in CATALOG)  # every skill appears in exactly one part


async def test_skills_of_one_group_stay_in_one_part():
    fake = judge("tax-3", SMALL)
    await SkillRouter(CATALOG, fake, FIT).decide(Turn("taxes"))
    for p in first_round(fake):
        groups = {n.split("-")[0] for n in p.options}
        for g in groups:
            assert all(f"{g}-{i}" in p.options for i in range(4))


async def test_option_count_limit_is_respected():
    fake = judge("docs-1", Limits(max_options=5))

    await SkillRouter(CATALOG, fake, Settings(max_candidates=3)).decide(Turn("statement"))

    assert all(len(p.options) <= 5 for p in picks(fake))


async def test_failed_part_is_split_and_asked_again_others_keep_working():
    fake = judge("tax-3", SMALL, fail_chunk_with="spend-0")

    d = await SkillRouter(CATALOG, fake, FIT).decide(Turn("taxes"))

    assert d.load == ("tax-3",) and d.trace.failure is None
    asked = [n for p in picks(fake) for n in p.options]
    assert "spend-0" in asked and "spend-3" in asked  # the refused part was asked again in halves


async def test_long_context_is_cut_to_its_end():
    fake = judge("spend-2", Limits())
    turn = Turn("and for April?", context="older " * 2000 + "the last message")
    await SkillRouter(CATALOG, fake, Settings(context_chars=100)).decide(turn)

    context = fake.calls[0][0]["context"]

    assert len(context) <= 100 and context.endswith("the last message")


def test_settings_that_cannot_fit_the_judge_fail_at_once():
    with pytest.raises(JudgeMisconfigured, match="limit"):
        SkillRouter(CATALOG, judge("x", SMALL), Settings(head_chars=5000))


async def test_search_works_on_a_split_catalog():
    fake = judge("tax-3", SMALL)

    found = await SkillRouter(CATALOG, fake, FIT).search("taxes", limit=2)

    assert found[0].name == "tax-3"


async def test_fitting_catalog_is_also_asked_again_in_halves_if_the_provider_refuses_the_size():
    fake = judge("tax-3", Limits(), fail_chunk_with="docs-0")  # the whole catalog in one call, and a refusal

    d = await SkillRouter(CATALOG, fake).decide(Turn("taxes"))

    assert d.load == ("tax-3",) and d.trace.failure is None
