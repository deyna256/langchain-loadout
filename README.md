<div align="center">

<img src="https://raw.githubusercontent.com/deyna256/langchain-loadout/main/docs/assets/banner.svg" alt="Loadout — selects the skills each turn needs from a larger catalog" width="100%">

<p><strong>Per-turn skill selection for LangChain and deepagents agents: the model sees the few skills
it needs, not a catalog of hundreds.</strong></p>

[![CI](https://github.com/deyna256/langchain-loadout/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deyna256/langchain-loadout/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-loadout)](https://pypi.org/project/langchain-loadout/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdeyna256%2Flangchain-loadout%2Fmain%2Fpyproject.toml)](pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/deyna256/langchain-loadout)](LICENSE)

[Quick start](#quick-start) · [Results](#results) · [Documentation](#documentation) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why Loadout

Large skill catalogs take up context on every model call. Loadout ranks and verifies skills for each
user turn, then loads the relevant instructions or suggests candidates. The agent can also search the
catalog with `find_skill`.

<table>
<tr>
<td width="50%" valign="top">

**Focused context**<br>
Confidence determines what gets loaded or suggested. Every threshold is configurable.

</td>
<td width="50%" valign="top">

**Independent turns**<br>
No skill state to carry between turns. No checkpointer or additional storage required.

</td>
</tr>
<tr>
<td valign="top">

**Your choice of judge**<br>
Use the included [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) adapter
or implement the small `Judge` interface.

</td>
<td valign="top">

**Graceful fallback**<br>
Decision timeouts and transient judge failures restore the full catalog.
Configuration errors surface explicitly.

</td>
</tr>
</table>

## Quick start

**1. Install.** Requires Python 3.11+. The `jev` extra includes the judge adapter and its SDK.

```sh
pip install "langchain-loadout[jev]"
```

**2. Connect your skills.** Place them in `./skills/<name>/SKILL.md` with `name` and `description`
in YAML front matter. Set `TYPESAFE_API_KEY` and your model provider's credentials
(`ANTHROPIC_API_KEY` for this example).

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from langchain_loadout.langchain import LoadoutSkillsMiddleware
from langchain_loadout.providers.jev import JevJudge

backend = FilesystemBackend(root_dir=".", virtual_mode=True)
loadout = LoadoutSkillsMiddleware(backend=backend, sources=["/skills/"], judge=JevJudge())

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    backend=backend,
    skills=["/skills/"],
    middleware=[loadout],
)
await agent.ainvoke({"messages": [{"role": "user", "content": "I need a statement for the embassy"}]})
```

Run the example in an async context. Selection runs on `ainvoke` and `astream`; synchronous calls use
the ordinary skills middleware. Tune thresholds and questions through `Settings`.

The public API is evolving: minor releases may introduce breaking changes while the version is `0.x`.

## Results

Latest benchmark on a bank-statement assistant using deepagents and **langchain-loadout 0.1.0 from
PyPI**: 236 skills, **50 conversations × 5 turns per variant** (250 turns each).

| Metric | With Loadout | Full catalog |
|---|---|---|
| Correct skill selected | **82%** | 61% |
| Skills section per model call | **5,174 characters** | 89,150 characters |
| Input tokens per turn | **34,630** | 113,541 |
| Answer accuracy | 88% | 86% |
| Cost per turn | $0.0056 | **$0.0039** |

The skills section was **17.2× smaller**, with **3.3× fewer input tokens**, but cost per turn was
**44% higher**. Smaller prompts do not necessarily mean lower cost when cache reuse changes.

These results use generated data, one judge and one agent model; the two-point accuracy difference
alone does not establish an accuracy improvement. Fit thresholds to your own data.
[Earlier measurements and design trade-offs →](docs/design.md)

## Documentation

| Read | Covers |
|---|---|
| [How it works](docs/design.md) | Selection flow, judge interface, settings and trade-offs |
| [Development guide](docs/development.md) | Architecture, Python conventions and testing |
| [Contributing](CONTRIBUTING.md) | Local setup, checks, issues and pull requests |
| [Changelog](CHANGELOG.md) | Release history |

## License

[MIT](LICENSE) © 2026 Ivan Deyna
