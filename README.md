# langchain-loadout

<p><strong>Per-turn skill selection for LangChain and deepagents agents: the model sees the few skills
it needs, not a catalog of hundreds.</strong></p>

[![CI](https://github.com/deyna256/langchain-loadout/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deyna256/langchain-loadout/actions/workflows/ci.yml)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdeyna256%2Flangchain-loadout%2Fmain%2Fpyproject.toml)](pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/deyna256/langchain-loadout)](LICENSE)

[What it gives](#what-it-gives) · [Limits](#limits) · [How it works](docs/design.md) ·
[Contributing](CONTRIBUTING.md)

---

> [!NOTE]
> Not released yet. The library works and is measured, but the public API is still moving and there is
> no package on PyPI.

An agent with hundreds of skills carries every name and description in its system prompt, on every
model call. Loadout decides each turn which skills matter and shows the model only those.

- **Cheap questions first.** Is a skill needed at all, and is what is already loaded enough — asked
  without the catalog, in about 0.35 s. Ranking and verification run only when a new skill is needed.
- **The logic is Loadout's, the judge is yours.** The questions are simple — "pick one", "yes or no" —
  and go through a single port. A ready adapter ships for
  [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- **Confidence decides what the agent sees.** High: the skill's instructions go straight into the
  request. Medium: two or three candidates. Low: nothing, and the model can still call `find_skill`.
  Every threshold is a setting.
- **A failure does not break the agent.** Loadout wraps the ordinary skills middleware. If it times out
  or errors, the agent gets the full list, exactly as it would without Loadout.

## What it gives

Measured on a testbed — a bank-statement assistant with a catalog of 236 skills, an agent on
deepagents, 50 conversations of 5 turns each:

| | with Loadout | full catalog in the prompt |
|---|---|---|
| correct answer to the user | 86% | 86% |
| the right skill was taken | **86%** | 57% |
| skills section of the prompt | **5,648 characters** | 89,150 characters |
| input tokens per turn | **34,131** | 111,864 |

When the agent has tools and can work the answer out for itself, the skill barely affects whether the
answer is right. What Loadout delivers consistently is context and a predictable skill choice. The
reasoning behind the design is in [docs/design.md](docs/design.md).

## Limits

- **It is not an accuracy feature.** Where a skill only restates what the model could work out, the
  answer is the same either way.
- **A decision costs about 3 s on a catalog of 236 skills.** Two seconds is reachable on a catalog of
  about a hundred, or on a turn that continues a topic.
- **A turn costs slightly more, not less.** The full catalog is identical every message and caches
  well; the Loadout prompt changes every turn and does not.
- **Thresholds have to be fitted on your own data**, and the library has no procedure for that yet.
- **Everything above was measured on generated data**, with one judge and one agent model.

The reasoning behind each of these is in [docs/design.md](docs/design.md#known-limits).

## Development

```sh
uv sync
just test     # the test suite; no network needed
just lint     # formatting, style and import order
just type     # types
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues, branches, commits and reviews, and
[docs/development.md](docs/development.md) for the coding rules.

## License

[MIT](LICENSE)
