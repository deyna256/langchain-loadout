"""The framework-free core: the data, the judge port and the router that decides a turn.

Nothing here imports langchain, deepagents or a provider SDK. Its public names are re-exported from
`langchain_skill_router`, which is where callers should import them from.
"""
