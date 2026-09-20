"""Loadout for deepagents: a wrapper around the ordinary skills middleware (`SkillsMiddleware`).

Skills are still discovered by the ordinary middleware, through `state["skills_metadata"]`. On each new
user message Loadout decides which of them are needed, and the model sees only those; the text of the
loaded ones goes straight into the system message. On any failure the request passes to the ordinary
middleware untouched, so the model sees the full list exactly as it would without Loadout.

Wiring: `create_deep_agent(..., skills=[...], middleware=[LoadoutSkillsMiddleware(...)])`. The wrapper
carries the built-in middleware's name and takes its place. The optimisation applies when the agent is
run asynchronously (`ainvoke`, `astream`); a synchronous run takes the ordinary path.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired, cast

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.skills import SkillMetadata, SkillsMiddleware, SkillsState
from langchain.agents.middleware.types import ModelRequest, ModelResponse, PrivateStateAttr
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

from langchain_loadout.core.judge import Judge
from langchain_loadout.core.router import SkillRouter
from langchain_loadout.core.types import DEFAULTS, Decision, Settings, Skill, Turn

PROMPT = """## Skills System

Skills picked for the current request (only these are listed; the full catalog is larger):

{skills_list}

{loaded}If none of the listed skills fits the task, call `find_skill` with a short description of what you need.
To use a listed skill that is not loaded below, read its SKILL.md with `read_file` (pass `limit=1000`)."""

LOADED = "**Loaded skill instructions — follow them:**\n\n{texts}\n\n"


class LoadoutState(SkillsState):
    loadout_turn: NotRequired[Annotated[str, PrivateStateAttr]]  # id of the user message this decision was made for
    loadout_failed: NotRequired[Annotated[bool, PrivateStateAttr]]  # a failure means the ordinary full list
    loadout_loaded: NotRequired[Annotated[list[str], PrivateStateAttr]]  # the text of these skills goes into the request
    loadout_suggest: NotRequired[Annotated[list[str], PrivateStateAttr]]  # these are listed only


def recent_context(messages: Sequence[AnyMessage], limit: int = 6) -> str:
    """The default context: the messages leading up to the current user message."""
    lines = [f"{m.type}: {str(m.content)[:500]}" for m in messages[-limit:] if str(m.content).strip()]
    return "\n".join(lines)


class LoadoutSkillsMiddleware(SkillsMiddleware):
    state_schema = LoadoutState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        sources: Sequence[str],
        judge: Judge,
        settings: Settings = DEFAULTS,
        catalog_hint: str = "",
        context: Callable[[Sequence[AnyMessage]], str] = recent_context,
        on_decision: Callable[[Decision], None] | None = None,
    ) -> None:
        super().__init__(backend=backend, sources=sources)
        self.judge, self.settings, self.context = judge, settings, context
        self.on_decision = on_decision  # every decision's trace, for logs, metrics and measurement
        self._router: SkillRouter | None = None
        self._router_for: tuple[str, ...] = ()
        self._paths: dict[str, str] = {}
        self.tools: list[BaseTool] = [self._find_skill_tool(catalog_hint)]

    @property
    def name(self) -> str:
        return "SkillsMiddleware"  # takes the place of the built-in deepagents middleware

    # --- the decision: once per new user message ----------------------------------------------------

    async def abefore_model(self, state: SkillsState, runtime: Any) -> dict[str, Any] | None:
        # The base class fixes this parameter to SkillsState, while `state_schema = LoadoutState` is what
        # the graph actually builds, so the narrowing has to be stated here rather than in the signature.
        ours = cast(LoadoutState, state)
        messages = ours["messages"]
        last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
        if last is None:
            return None
        turn_id = messages[last].id or str(last)
        if ours.get("loadout_turn") == turn_id:
            return None  # this turn already has a decision
        router = self._router_from(ours.get("skills_metadata", []))
        loaded = tuple(ours.get("loadout_loaded", []))
        turn = Turn(request=str(messages[last].content), context=self.context(messages[:last]), loaded=loaded)
        decision = await router.decide(turn)
        if self.on_decision:
            self.on_decision(decision)
        if decision.trace.failure:
            return {"loadout_turn": turn_id, "loadout_failed": True}
        return {
            "loadout_turn": turn_id,
            "loadout_failed": False,
            "loadout_loaded": [*decision.keep, *decision.load],
            "loadout_suggest": list(decision.suggest),
        }

    # --- applying it: what the model sees -----------------------------------------------------------

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        state = request.state
        if "loadout_turn" not in state or state.get("loadout_failed"):
            return await super().awrap_model_call(request, handler)  # the ordinary path: the full list
        loaded = state.get("loadout_loaded", [])
        picked = set(loaded) | set(state.get("loadout_suggest", []))
        listed: list[SkillMetadata] = [m for m in state.get("skills_metadata", []) if m["name"] in picked]
        texts = [await self._text(name) for name in loaded if name in self._paths]
        section = PROMPT.format(
            skills_list=self._format_skills_list(listed) if listed else "(nothing picked for this request)",
            loaded=LOADED.format(texts="\n\n---\n\n".join(texts)) if texts else "",
        )
        return await handler(request.override(system_message=append_to_system_message(request.system_message, section)))

    # --- catalog and search -------------------------------------------------------------------------

    def _router_from(self, metadata: list[SkillMetadata]) -> SkillRouter:
        names = tuple(m["name"] for m in metadata)
        if self._router is None or names != self._router_for:
            self._paths = {m["name"]: m["path"] for m in metadata}
            catalog = [Skill(m["name"], m["description"], self._reader(m["name"])) for m in metadata]
            self._router, self._router_for = SkillRouter(catalog, self.judge, self.settings), names
        return self._router

    def _reader(self, name: str) -> Callable[[], Awaitable[str]]:
        async def read() -> str:
            return await self._text(name)

        return read

    async def _text(self, name: str) -> str:
        """Read a skill's instructions from the backend. Deliberately not cached: an agent that runs for
        days would otherwise keep serving the text a SKILL.md had when it first read it."""
        [response] = await self._backend.adownload_files([self._paths[name]])
        if response.error or response.content is None:
            raise OSError(f"could not read {self._paths[name]}: {response.error}")
        return response.content.decode()

    def _find_skill_tool(self, catalog_hint: str) -> BaseTool:
        async def find_skill(query: str) -> str:
            if self._router is None:
                return "Skill catalog is not loaded yet."
            found = await self._router.search(query)
            if not found:
                return "No matching skill found."
            return "\n".join(
                f"- **{s.name}**: {s.description}\n  -> Read `{self._paths[s.name]}` for full instructions" for s in found
            )

        description = (f"{catalog_hint} " if catalog_hint else "") + (
            "Find skills (step-by-step instructions) for a task that the skills listed in the system prompt don't cover. "
            "Pass a short description of the task; returns the best matching skills with paths to read."
        )
        return StructuredTool.from_function(coroutine=find_skill, name="find_skill", description=description)
