"""Check a judge adapter against the contract the router relies on.

An adapter author runs this against their provider:

    from langchain_loadout.testing import check_judge

    async def test_my_adapter():
        await check_judge(MyJudge())

It makes one real call, so it costs whatever the provider charges for one call.
"""

from typing import NoReturn

from langchain_loadout.core.judge import Judge, Limits, Pick, YesNo

STATE = {"request": "Show me what I spent last month, broken down by category."}

SKILLS = {
    "spending-by-category": "Break the user's spending down by category over a period.",
    "card-limits": "Explain the limits set on the user's card.",
    "visa-statement": "Produce a bank statement formatted for a visa application.",
}

QUESTIONS: dict[str, Pick | YesNo] = {
    "pick": Pick("Which skill's instructions would help answer `request` best?", SKILLS),
    "about-spending": YesNo("Is `request` about money the user has already spent?"),
    "about-transfer": YesNo("Is `request` asking to transfer money to someone?"),
}


def _fail(what: str) -> NoReturn:
    raise AssertionError(f"judge does not meet the contract: {what}")


def _require(ok: bool, what: str) -> None:
    if not ok:
        _fail(what)


def _probability(value: object, what: str) -> float:
    if not isinstance(value, int | float):
        _fail(f"{what} is {type(value).__name__}, expected a number")
    number = float(value)
    _require(0.0 <= number <= 1.0, f"{what} is {number}, expected a probability within [0, 1]")
    return number


async def check_judge(judge: Judge) -> None:
    """Raise `AssertionError` unless the adapter meets every requirement of the `Judge` contract."""
    limits = getattr(judge, "limits", None)
    if not isinstance(limits, Limits):
        _fail("`limits` is not declared; use `Limits()` if the provider has none")
    _require(limits.max_tokens > 0, f"limits.max_tokens is {limits.max_tokens}, expected a positive number")
    _require(limits.max_options > 1, f"limits.max_options is {limits.max_options}, expected more than one option")
    _require(limits.tokens_per_char > 0, f"limits.tokens_per_char is {limits.tokens_per_char}, expected a positive number")
    timeout = getattr(judge, "timeout", None)
    _require(timeout is None or timeout > 0, f"timeout is {timeout}, expected a positive number of seconds or None")

    answers = await judge.ask(STATE, QUESTIONS)

    _require(set(answers) == set(QUESTIONS), f"one call answered {sorted(answers)}, expected {sorted(QUESTIONS)}")

    picked = answers["pick"].probabilities
    _require(
        set(picked) == set(SKILLS),
        f"a Pick answered about {sorted(picked)}, expected a probability for every option: {sorted(SKILLS)}",
    )
    for name, value in picked.items():
        _probability(value, f"the probability of option {name!r}")
    best = max(picked, key=picked.__getitem__)
    _require(best == "spending-by-category", f"the best option was {best!r}, expected 'spending-by-category'")

    spending = _probability(answers["about-spending"].probabilities.get("yes"), "the answer to a YesNo question")
    transfer = _probability(answers["about-transfer"].probabilities.get("yes"), "the answer to a YesNo question")
    _require(
        spending > transfer,
        f"an obvious yes scored {spending} and an obvious no scored {transfer}; "
        "probabilities have to be calibrated, because the router compares them against thresholds",
    )
