"""Integration with LangChain agents. Importing this package pulls in langchain and deepagents."""

from langchain_loadout.langchain.middleware import LoadoutSkillsMiddleware, recent_context

__all__ = ["LoadoutSkillsMiddleware", "recent_context"]
