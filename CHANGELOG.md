# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.x` the public API
may change in a minor release.

Add the entry for a change in the same commit as the change itself.

## [Unreleased]

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

[Unreleased]: https://github.com/deyna256/langchain-loadout/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deyna256/langchain-loadout/releases/tag/v0.1.0
