"""The judge port: simple questions in our own vocabulary, which an adapter translates for a provider."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Pick:
    """Pick one option. `options` maps a key to its description."""

    instructions: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class YesNo:
    """Yes or no."""

    instructions: str


@dataclass(frozen=True)
class Answer:
    """For `Pick`, option to probability. For `YesNo`, {"yes": probability of yes}."""

    probabilities: Mapping[str, float]

    @property
    def yes(self) -> float:
        return self.probabilities["yes"]


@dataclass(frozen=True)
class Limits:
    """A provider's limits for one call. The defaults mean the provider declared none."""

    max_tokens: int = 1_000_000  # the state plus the longest question
    max_options: int = 1_000_000  # options in a single choice
    tokens_per_char: float = 1.0  # upper estimate of tokens per character of text


class LoadoutError(Exception):
    """Base for every error this library raises."""


class JudgeUnavailable(LoadoutError):
    """The judge could not answer this time: a network problem, a timeout, a provider error.

    The router treats this as a failed turn and falls back: the agent gets the full catalog, exactly as
    it would without Loadout.
    """


class JudgeMisconfigured(LoadoutError):
    """The judge cannot work at all: missing or rejected credentials, or an adapter that does not meet
    the contract.

    Retrying does not help, so this is never swallowed. It surfaces to the application instead of
    turning into a fallback that hides a broken setup for the life of the process.
    """


class Judge(Protocol):
    """What an adapter has to provide for the router to work.

    `ask` answers every question in one call: that is where the cost saving comes from, and an adapter
    for a provider that answers one at a time has to fan out internally. The state is a mapping of
    JSON-serialisable values. A `Pick` is answered with a probability for every option, not just the
    winner, because the router ranks candidates from that distribution, and the probabilities are
    compared against thresholds, so they have to be calibrated and within [0, 1].

    An adapter also declares `limits`, which the router uses to split a catalog that does not fit one
    call. Declare `Limits()` to say the provider has none; leaving it out is an error, because a silent
    default would split blind.

    Failures are reported as `JudgeUnavailable` when a retry could help and `JudgeMisconfigured` when it
    could not. Anything else an adapter raises is treated as unavailable.
    """

    limits: Limits

    async def ask(self, state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]: ...
