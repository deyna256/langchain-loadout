# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.x` the public API
may change in a minor release.

Add the entry for a change in the same commit as the change itself.

## [Unreleased]

## [0.2.0] - 2026-09-22

### Added

- `Settings.pick_question` and `Trace.picked`: verification now also asks which candidate is the right
  one, with the candidates' texts side by side, and records the answer.

### Changed

- Verification decides which skill to load by its pick among the candidates; the per-candidate "fits"
  answers only decide whether any is loaded (`load_at` against the best of them) and which are suggested
  (`suggest_at`). Ordered by "fits" alone, lookalike skills tied and the wrong one was often loaded. A
  trace recorded without a pick is ordered by the ranking.
- `Settings.max_load` defaults to 1: two lookalike skills loaded together gave worse answers than one.
- The loaded skill and the suggested ones now go in a message right after the user's request instead of
  the system message, which stays the same on every call. The provider's prompt cache keeps the
  conversation before the request; with the skills in the system message it was read again uncached on
  every turn. The message says the skill may be ignored if it does not fit, following TypeSafe's
  skill-suggestion cookbook.
- `recent_context`, the default context for the judge, now carries what the user asked and the agent
  answered, without tool calls and their results, which pushed the previous request out of the window.
- Settings that cannot fit verification into one call are rejected at construction with the pick's size
  counted in: every candidate's text now goes out twice.

- Clarified README positioning: a deepagents extension built on LangChain middleware, with an
  independently usable routing core.
- Updated README benchmark results for the PyPI 0.1.0 run, including per-turn costs and sample size.
- PyPI publication is now started manually through Publish with an existing release tag. Release only
  creates the GitHub release, keeping publication attestations tied to the configured Trusted Publisher.

## [0.1.1] - 2026-09-21

### Fixed

- Instruction read errors now fall back to the ordinary full skills catalog for that model call,
  without injecting partially loaded instructions or retrying model errors.
- Skill selection, instruction loading and `find_skill` now use the current execution's catalog,
  including updated descriptions and paths, without sharing a cached router between concurrent runs.

### Changed

- Refreshed the README with a visual overview, a compact quick start and links to detailed documentation.

## [0.1.0] - 2026-09-20

The first release.

### Added

- `SkillRouter`, which decides on each turn which skills an agent should see: ranking over the catalog
  with "is a skill needed at all" asked alongside it, then verification of the candidates against their
  own text. Every threshold and every question is a `Settings` field. A turn is decided on its own, so
  the library needs no checkpointer and keeps nothing between turns.
- `LoadoutSkillsMiddleware` for deepagents agents, which takes the place of `SkillsMiddleware` and
  falls back to it whenever a decision fails.
- The `Judge` port, with `JevJudge` behind the optional `jev` extra, so the library depends on no
  provider SDK by default.
- `langchain_loadout.testing`: `check_judge` runs an adapter against the contract in one call, and
  `ScriptedJudge` answers from a script so an agent can be tested without a provider.
- An error taxonomy: `JudgeUnavailable` falls back to the full catalog, `JudgeMisconfigured` surfaces
  instead of hiding a broken setup, and both derive from `LoadoutError`.
- A public API: everything listed in `langchain_loadout.__all__`, importable from the package root and
  pulling in neither a framework nor a provider SDK.

[Unreleased]: https://github.com/deyna256/langchain-loadout/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/deyna256/langchain-loadout/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/deyna256/langchain-loadout/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/deyna256/langchain-loadout/releases/tag/v0.1.0
