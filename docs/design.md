# How Loadout works

## The problem

An agent with tens or hundreds of skills keeps the name and description of every one of them in the
system prompt, on every model call. The list grows with the catalog, takes up context and distracts the
model: the more options there are, the worse it chooses — Anthropic reports a marked drop past 30–50
tools, and on a catalog of 182 skills an agent loaded the wrong skill in 16.8% of requests. Loadout
decides on each turn which skills are needed, and the model sees only those, so that it works on the
user's request rather than on the catalog.

## What the measurements settled

This section records the earlier measurements that informed the design. The latest benchmark of the
PyPI 0.1.0 release is reported in the [README](../README.md#results).

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

1. **Loadout collects nothing and remembers nothing.** The catalog and the turn's context come from the
   product or its framework, and each turn is decided on its own. Nothing is carried over from the turn
   before it, so the library needs no checkpointer and no store of its own.
2. **The judge is a fast classifier that returns probabilities** (Jev today). The logic — which
   questions to ask and how to read the answers — stays in Loadout; a provider is plugged in through
   the `Judge` port, which speaks in "pick one" and "yes or no".
3. **A decision is a loadout for the turn:** load (the instruction text goes straight into the
   request) and suggest (two or three candidates the model chooses from).
4. **Every threshold is a product setting** (`Settings`), fitted on the product's own data.
5. **A Loadout failure does not break the conversation:** the worst outcome is that the agent gets its
   skills the ordinary way.

## The turn

```
user message
  │
  ├─ 1. "Is a skill needed at all" (~400 tokens), asked alongside the ranking rather than before it
  │     └─ answered low → the decision is ready, and nothing is loaded
  │
  ├─ 2. Ranking over the catalog: pick by name and description → the best candidates with probabilities
  │     (over the provider's limit → parts in parallel, then a merge)
  │     └─ `skip_verify_at` is set and the first candidate is above it → load without verifying
  │
  └─ 3. Verification, one call over the heads of the candidates' texts:
        "which of these is the right one?" — a pick with the texts side by side — settles which skill,
        and "do these instructions do what the request asks?" for each on its own settles whether any
        → thresholds: load the pick / suggest / nothing
```

**Why it is shaped this way**, from the testbed measurements:

- Verification asks two different things. Ordered by their independent yes/no answers alone, lookalikes
  tie (0.8 against 0.9, say) and the wrong one often comes first: over 263 conversation turns on the bank
  testbed the right skill was first by `fits` in 67% of turns, against 92% by the ranking over
  descriptions, and loading the two best side by side dropped the answer from 91% correct to 76%. So a
  pick over the candidates' texts decides which skill is loaded, one by default, and `fits` only decides
  whether any is — the split TypeSafe's skill-suggestion cookbook uses as well.
- A candidate's text goes in its own question rather than in the shared state, because otherwise its
  score depends on its neighbours in the same call — a shift of up to 0.46, against no more than 0.05
  this way — and the size of a call stops growing with the number of candidates.
- "Is a skill needed at all" travels with the ranking instead of preceding it. Asked first it ends
  about one turn in fifty on its own and costs every other turn a round trip; asked alongside, it costs
  neither. At a threshold of 0.05 it errs towards "not needed" in 0-0.7% of requests.

## The interface

The core is imported from the package root, and the parts that carry a dependency from their own
modules:

```python
from langchain_loadout import Settings, SkillRouter, Turn  # core: no framework, no provider
from langchain_loadout.langchain import LoadoutSkillsMiddleware  # pulls in langchain and deepagents
from langchain_loadout.providers.jev import JevJudge  # pulls in typesafe-sdk

router = SkillRouter(catalog, judge, settings)
decision = await router.decide(Turn(request, context))  # load / suggest + trace
found = await router.search(query)  # for the find_skill tool
```

`Settings` carries three kinds of knob. **How much:** `max_candidates`, `max_load`, `head_chars`,
`request_chars`, `context_chars`, `budget_share`, `timeout`. **Thresholds:** `need_at`, `load_at`,
`suggest_at` and `skip_verify_at`, which is off by default. **The questions the judge is asked:**
`need_question`, `rank_question`, `pick_question` and `fits_question` — the defaults are written for a general
assistant, and wording that names the product's own domain separates better. `Trace` carries the
ranking probabilities, the answer to every question, where the decision ended, how long it took and
what failed.

Each threshold is compared against the trace field of the same name: `need_at` against `trace.need`,
`load_at` against the best of `trace.fits`, `suggest_at` against each candidate's. Which candidate is
loaded follows `trace.picked`, verification's pick; a trace recorded without one follows the ranking.

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
decides, and the model sees only the picked skills. The layout follows the provider's prompt cache, which
matches a request from its start up to the first changed character: the system message gets a section
that is the same on every call, and the turn's skills — the loaded one's instructions, the others by name
and description — go in a message right after the user's request. What the turn picked changes only what
follows the request, and the conversation before it stays cached. The message is added to a copy of the
request, not to the agent's state, so nothing the user sees changes and nothing has to survive until the
next turn. On failure the request goes to the ordinary middleware with the full list.

The judge sees the turn's context as the product assembles it; the default, `recent_context`, is the
last few things the user asked and the agent answered, without tool calls and their results, which would
otherwise crowd the previous request out of the window. `find_skill(query)` covers the case where the model needs a skill that
is not in the list. The optimisation applies when the agent is run asynchronously.

## Known limits

- **A turn's skills are not cached across turns.** They follow the request and are not kept in the
  conversation, so the next turn reads its own skills, and the tail of the previous turn, uncached. With
  the skills in the system message the whole conversation was read uncached on each turn's first call
  instead; keeping them in the conversation would need the "already loaded" knowledge the next limit
  explains the library does not rely on.
- **Two seconds is out of reach on a large catalog.** On 236 skills the catalog splits into three
  parts, so a full pass costs about 58k tokens and about 3 s; only the cheap decisions land inside two
  seconds. The size estimate is part of the problem: 0.8 tokens per character is assumed against about
  0.68 measured, which causes unnecessary splitting.
- **Thresholds have to be fitted per product**, and there is no procedure in the library for it yet.
  `apply_policy` is kept separate from the calls so traces can be refitted without paying the provider
  again, but the workflow around that is missing.
- **Every turn pays for a full pass.** Loadout deliberately carries nothing between turns, so a
  continuation of a topic costs the same as a new one: 3.1 s against the 1.6 s it cost while the
  library kept track of what was loaded. Removing that also cost about three points of skill-choice
  accuracy on continuations, measured by replaying the recorded answers.

  It was removed because it could not be made reliable. Graph state does not survive an invocation
  without a checkpointer, a private field cannot even be passed back in, and the message history is no
  better because middleware clips and offloads tool results. The errors are asymmetric: believing a
  skill is loaded when it is not makes the agent answer without the procedure, silently, while not
  knowing about a loaded skill only costs a pass over the catalog. Bringing it back needs a carrier the
  library is willing to ask an application for.

Open work is tracked in issues.

## How this differs from what already exists

| approach | how it chooses | what is missing |
|---|---|---|
| progressive disclosure (Agent Skills, the deepagents `SkillsMiddleware`) | every skill's name and description in the prompt, the text on request | the list grows with the catalog, and the choice is still the model's |
| tool search / `defer_loading` ([Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)) | the model searches for itself with regex or BM25 | the model has to realise it should search; an extra step |
| `LLMToolSelectorMiddleware` ([LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in)) | a separate LLM call on every step | slow and expensive |
| vector retrieval ([RAG-MCP](https://arxiv.org/abs/2505.03275), [dynamic-tools](https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware)) | similarity between descriptions | no answer to "is a skill needed at all", no confidence; ordinary retrievers are poor at finding tools ([ToolRet](https://aclanthology.org/2025.findings-acl.1258/)) |
| skill suggestion on Jev ([TypeSafe](https://docs.typesafe.ai/cookbooks/skill_suggestion.md)) | a classifier plus candidate verification | the whole catalog stays in the prompt |
