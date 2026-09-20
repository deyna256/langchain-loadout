"""The data the router exchanges with the product. Knows nothing about providers or frameworks."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    """A skill from the product's catalog. Loadout does not discover skills; it is handed them."""

    name: str
    description: str
    read: Callable[[], Awaitable[str]]  # the full instructions; called for candidates only
    group: str | None = None


@dataclass(frozen=True)
class Turn:
    """One turn of a conversation: everything a decision needs. The product assembles the context."""

    request: str
    context: str = ""
    loaded: tuple[str, ...] = ()


@dataclass(frozen=True)
class Settings:
    """Everything a product may want to tune."""

    max_candidates: int = 6  # how many candidates ranking passes to verification
    max_load: int = 2  # how many skills may be loaded in one turn
    load_at: float = 0.8  # "fits" at or above this: put the instructions in the request
    suggest_at: float = 0.4  # "fits" at or above this: offer the skill as a candidate
    keep_at: float = 0.5  # a loaded skill "still needed" at or above this: keep it
    need_at: float = 0.3  # "a skill is needed at all" below this: load nothing
    beyond_at: float = 0.5  # something is loaded and "anything beyond it" is below this: skip ranking
    skip_verify_at: float | None = None  # ranking this sure of its first candidate: load without verifying
    head_chars: int = 1500  # how much of a skill's text verification sees
    timeout: float = 2.0  # seconds for the whole decision
    request_chars: int = 2000  # how much of the request the judge sees (a user may paste a whole statement)
    context_chars: int = 4000  # how much of the context the judge sees, counted from the end
    budget_share: float = 0.85  # share of the provider's limit to use, leaving room for estimation error
    # Every question the judge is asked. The defaults are written for a general assistant; wording that
    # names the product's own domain separates better (measured on bank requests, the need question went
    # from 0.80-0.97 against 0.01-0.02). The judge sees `request`, `context`, `loaded` and, when
    # something is loaded, `loaded_skills` in the state it is given.
    need_question: str = (
        "Does answering the user's request need a specialised skill (a documented procedure or work on the "
        "user's accounts, statements or documents) rather than a plain conversational reply?"
    )
    rank_question: str = "Which skill's instructions would help the assistant answer the user's request best?"
    # `fits_question` is formatted with the candidate's `name` and the head of its `text`.
    fits_question: str = (
        'Do these skill instructions do what the user asks for in `request`?\n\n<skill name="{name}">\n{text}\n</skill>'
    )
    # `still_question` is formatted with the loaded skill's `name`.
    still_question: str = "Is the already loaded skill `{name}` still needed to answer `request`, given `context`?"
    beyond_question: str = (
        "Does answering `request` need instructions beyond the already loaded skills (in `loaded_skills`), given `context`?"
    )


@dataclass(frozen=True)
class Trace:
    """What was asked, what came back and what broke. Recorded for measurement and debugging."""

    candidates: tuple[tuple[str, float], ...] = ()  # the best of the ranking, with probabilities, best first
    need: float | None = None
    beyond: float | None = None  # "anything needed beyond what is loaded", only when something is loaded
    fits: Mapping[str, float] = field(default_factory=dict)  # per candidate: does it do what the request asks
    still_needed: Mapping[str, float] = field(default_factory=dict)  # per loaded skill: is it still needed
    failure: str | None = None
    seconds: float = 0.0
    parts: int = 1  # how many parts the catalog was split into for ranking
    stage: str = ""  # where the decision ended: gate, skip (no verification) or verify


@dataclass(frozen=True)
class Decision:
    """What to do on this turn."""

    load: tuple[str, ...] = ()  # instructions go straight into the request
    suggest: tuple[str, ...] = ()  # listed as candidates for the model to choose from
    keep: tuple[str, ...] = ()  # loaded skills that stay
    drop: tuple[str, ...] = ()  # loaded skills that go
    trace: Trace = field(default_factory=Trace)


DEFAULTS = Settings()  # Settings is frozen, so one shared instance is enough
