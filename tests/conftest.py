"""Building blocks the tests share."""

from langchain_skill_router import Skill


def skill(name: str, *, group: str | None = None, text: str | None = None, description: str | None = None) -> Skill:
    """A catalog entry whose instructions are read on demand, the way the router expects them."""

    async def read() -> str:
        return text if text is not None else f"# {name}\nInstructions for {name}."

    return Skill(name=name, description=description or f"Description of {name}", read=read, group=group)
