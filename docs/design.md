# How Loadout works

## The problem

An agent with tens or hundreds of skills keeps the name and description of every one of them in the
system prompt, on every model call. The list grows with the catalog, takes up context and distracts the
model: the more options there are, the worse it chooses — Anthropic reports a marked drop past 30–50
tools, and on a catalog of 182 skills an agent loaded the wrong skill in 16.8% of requests. Loadout
decides on each turn which skills are needed, and the model sees only those, so that it works on the
user's request rather than on the catalog.

## What the measurements settled

Two measurements on a testbed — a Russian bank-statement assistant with a generated catalog and an
agent built on deepagents — set the terms the rest of this document is written in.

When the agent has tools and can work the answer out for itself, **the skill is not a precondition for
a correct answer**: 86% against 86% over 250 conversation turns. So Loadout is not an accuracy feature.
What it does deliver, consistently, is context and predictability: the skills section of the prompt is
16 times smaller (5.6k against 89k characters), total input is 3.3 times smaller (34k against 112k
tokens), and the right skill is taken in 86% of turns against 57%.

Accuracy does improve where a skill carries a domain rule rather than a procedure the model can infer —
88% against 64% on small-business questions, 71% against 52% on yes/no answers with a supporting
detail — and it drops where the agent follows an instruction too literally: 57% against 100% on date
questions, 61% against 89% on report requests.

Three things those numbers do not cover. Everything was measured on generated data: the statements, the
skills and the questions were written by a model to our rules. One judge and one agent model were used,
so how Loadout behaves with a different classifier is untested, even though the `Judge` port exists for
exactly that. And nothing has been measured on a catalog written by people, where description quality
is uneven.

## Decisions

1. **Loadout collects nothing itself.** The catalog and the turn's context come from the product or its
   framework. It scans nothing and remembers nothing: the set of loaded skills is held by the agent.
2. **The judge is a fast classifier that returns probabilities** (Jev today). The logic — which
   questions to ask and how to read the answers — stays in Loadout; a provider is plugged in through
   the `Judge` port, which speaks in "pick one" and "yes or no".
3. **A decision is a loadout for the turn:** load (the instruction text goes straight into the
   request), suggest (two or three candidates the model chooses from), drop (loaded skills that are no
   longer needed).
4. **Every threshold is a product setting** (`Settings`), fitted on the product's own data.
5. **A Loadout failure does not break the conversation:** the worst outcome is that the agent gets its
   skills the ordinary way.

## The turn

```
user message
  │
  ├─ 1. A cheap call with no catalog (~400 tokens, ~0.35 s):
  │     is a skill needed at all · is anything needed beyond what is loaded · is each loaded skill still needed
  │     └─ no new skill needed → the decision is ready
  │
  ├─ 2. Ranking over the catalog: pick by name and description → the best `shortlist` with probabilities
  │     (over the provider's limit → parts in parallel, then a merge)
  │     └─ `skip_verify_at` is set and the first candidate is above it → load without verifying
  │
  └─ 3. Verification: "do these instructions do what the request asks?" against the head of each
        candidate's text, each text carried in its own question → thresholds: load / suggest / nothing
```

**Why it is shaped this way**, from the testbed measurements:

- Verification earns its call: ranking alone puts the right skill first in 74.8% of requests, and
  verification raises that to 84.5%.
- A candidate's text goes in its own question rather than in the shared state, because otherwise its
  score depends on its neighbours in the same call — a shift of up to 0.46, against no more than 0.05
  this way — and the size of a call stops growing with the number of candidates.
- The cheap questions come first, so small talk and continuations of a topic never pay for a pass over
  the catalog. At a threshold of 0.05, "is a skill needed" errs towards "not needed" in 0–0.7% of
  requests.
- Each loaded skill gets its own question instead of one blanket "is anything else needed": that
  distinguishes "now export it to Excel", where the old skill still applies, from "and I also need a
  visa certificate", where it does not.

## The interface

The core is imported from the package root, and the parts that carry a dependency from their own
modules:

```python
from langchain_loadout import Settings, SkillRouter, Turn  # core: no framework, no provider
from langchain_loadout.langchain import LoadoutSkillsMiddleware  # pulls in langchain and deepagents
from langchain_loadout.providers.jev import JevJudge  # pulls in typesafe-sdk

router = SkillRouter(catalog, judge, settings)
decision = await router.decide(Turn(request, context, loaded))  # load / suggest / keep / drop + trace
found = await router.search(query)  # for the find_skill tool
```

`Settings` carries `shortlist`, `max_load`, `load_at`, `suggest_at`, `keep_at`, `need_at`, `beyond_at`,
`skip_verify_at` (off by default), `need_question`, `head_chars`, `request_chars`, `context_chars`,
`budget_share` and `timeout`. `Trace` carries the ranking probabilities, the answers to every question,
where the decision ended, how long it took and what failed.

## Plugging in a different judge

`Judge` is declared in the core, because the router is its consumer: an adapter satisfies the
interface, and the interface does not follow an adapter. What an adapter has to provide:

| requirement | why the router needs it |
|---|---|
| answer every question of a call in **one** request | this is where the cost saving comes from; a provider that answers one at a time fans out inside the adapter |
| a `Pick` returns a probability for **every** option | the router ranks candidates from that distribution, not from the winner alone |
| probabilities are calibrated and within [0, 1] | every setting is a threshold they are compared against |
| declare `limits` | the router splits a catalog that does not fit one call; `Limits()` says the provider has none |
| raise `JudgeUnavailable` or `JudgeMisconfigured` | the first falls back to the full catalog, the second surfaces |

The split matters. Rejected credentials are misconfiguration, because no retry fixes them and a
fallback would hide a broken setup for the life of the process. A rejected request is not, because the
router already answers that by splitting the catalog and asking again.

None of this is left to be read carefully:

```python
from langchain_loadout.testing import check_judge


async def test_my_adapter():
    await check_judge(MyJudge())
```

One real call checks the shape of the answers and that they are judgements rather than numbers: an
obvious yes has to outscore an obvious no, and the right option has to win the pick.

**Provider limits.** An adapter declares `Limits` — for Jev, 32k tokens for the state plus the longest
question, and 255 options. Loadout estimates the size of every call in advance, splits the catalog into
parts that fit, truncates the request and the context, and rejects settings that cannot fit at all. If
a part is refused anyway, it is asked again in halves.

## Inside the agent (deepagents)

`LoadoutSkillsMiddleware` takes the place of `SkillsMiddleware`, under the same name. Skills are still
discovered by the ordinary middleware through `state["skills_metadata"]`; on a new user message Loadout
decides, and the model sees a shortened list plus the text of the loaded skills. Only a copy of the
request is shortened — the agent's state is not touched. On failure the request goes to the ordinary
middleware with the full list. `find_skill(query)` covers the case where the model needs a skill that
is not in the list. The optimisation applies when the agent is run asynchronously.

## Known limits

- **The Loadout prompt is not cached.** The full catalog is identical in every message and a gateway
  caches 94% of it; the Loadout prompt changes every turn and caches at 68%, which made a turn cost
  $0.0049 against $0.0039. Keeping the selected skills at the end of the prompt, so the unchanging head
  stays cacheable, should fix it.
- **Two seconds is out of reach on a large catalog.** On 236 skills the catalog splits into three
  parts, so a full pass costs about 58k tokens and about 3 s; only the cheap decisions land inside two
  seconds. The size estimate is part of the problem: 0.8 tokens per character is assumed against about
  0.68 measured, which causes unnecessary splitting.
- **Thresholds have to be fitted per product**, and there is no procedure in the library for it yet.
  `apply_policy` is kept separate from the calls so traces can be refitted without paying the provider
  again, but the workflow around that is missing.
- **Cross-turn memory needs a carrier the library does not have.** Keeping and dropping skills is what
  makes a continuation cheap — 1.6 s against 3.1 s — but it has to know which skills are loaded. Graph
  state does not survive an invocation without a checkpointer, a private field cannot even be passed
  back in, and the message history is unreliable because middleware clips and offloads tool results.
  The errors are asymmetric: believing a skill is loaded when it is not makes the agent answer without
  the procedure, silently, while not knowing about a loaded skill only costs a pass over the catalog.

Open work is tracked in issues.

## How this differs from what already exists

| approach | how it chooses | what is missing |
|---|---|---|
| progressive disclosure (Agent Skills, the deepagents `SkillsMiddleware`) | every skill's name and description in the prompt, the text on request | the list grows with the catalog, and the choice is still the model's |
| tool search / `defer_loading` ([Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)) | the model searches for itself with regex or BM25 | the model has to realise it should search; an extra step |
| `LLMToolSelectorMiddleware` ([LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in)) | a separate LLM call on every step | slow and expensive |
| vector retrieval ([RAG-MCP](https://arxiv.org/abs/2505.03275), [dynamic-tools](https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware)) | similarity between descriptions | no answer to "is a skill needed at all", no confidence; ordinary retrievers are poor at finding tools ([ToolRet](https://aclanthology.org/2025.findings-acl.1258/)) |
| skill suggestion on Jev ([TypeSafe](https://docs.typesafe.ai/cookbooks/skill_suggestion.md)) | a classifier plus candidate verification | the whole catalog stays in the prompt |
