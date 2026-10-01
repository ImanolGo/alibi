"""MCP server: the game world as tools for any MCP client.

Tool descriptions matter — they are prompts too. An LLM detective learns what it
can do from these strings, so keep them short, accurate and honest about cost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .case import Case
from .game import ActionResult, Game

log = logging.getLogger(__name__)


def build_server(game: Game, *, name: str = "alibi") -> MCPServer:
    """Expose one :class:`Game` as a set of MCP tools."""
    server = MCPServer(
        name=name,
        instructions=(
            "You are the detective in a cosy murder mystery. Question suspects, "
            "search rooms and examine clues, then accuse one suspect of the murder."
        ),
    )

    @server.tool()
    def case_summary() -> str:
        """The case in brief: the victim and the suspects you can question."""
        return game.case_summary().message

    @server.tool()
    def status() -> str:
        """How many actions remain, what you have found, and whether the game is over."""
        return game.status().message

    @server.tool()
    def list_rooms() -> str:
        """List the rooms you can search, with the id you must pass to search_room."""
        return game.list_rooms().message

    @server.tool()
    def search_room(room_id: str) -> str:
        """Search a room for clues (costs 1 action). Returns what you notice there."""
        return _text(game.search_room(room_id))

    @server.tool()
    def inspect(clue_id: str) -> str:
        """Examine a clue you have already found (costs 1 action); may implicate someone."""
        return _text(game.inspect(clue_id))

    @server.tool()
    def question(suspect_id: str, text: str) -> str:
        """Ask a suspect a question (costs 1 action). They stay in character and may lie."""
        return _text(game.question(suspect_id, text))

    @server.tool()
    def lie_detector() -> str:
        """Use the lie detector (free, max 2). Currently unavailable until M5."""
        return _text(game.lie_detector())

    @server.tool()
    def accuse(suspect_id: str, motive: str | None = None) -> str:
        """Name the murderer (ends the game). Optionally give a motive."""
        return _text(game.accuse(suspect_id, motive))

    return server


def _text(result: ActionResult) -> str:
    return ("" if result.ok else "ERROR: ") + result.message


def serve(
    case_path: str | None = None,
    *,
    game_id: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the MCP server. ``stdio`` for a local client; an HTTP transport for a
    deployed service. Resumes a saved game if one exists, else starts fresh."""
    import os

    from .db import get_engine, init_db
    from .game import SqlGameStore, load_or_create_game

    resolved = case_path or os.environ.get("ALIBI_CASE") or _newest_case()
    if not resolved:
        raise SystemExit(
            "No case found. Generate one with `alibi case`, pass --case, or set ALIBI_CASE."
        )
    case = Case.model_validate_json(Path(resolved).read_text(encoding="utf-8"))
    init_db(get_engine())
    store = SqlGameStore()
    game = load_or_create_game(case, game_id=game_id or case.id, store=store)
    server = build_server(game)

    if transport == "streamable-http":
        server.run(transport="streamable-http", host=host, port=port)
    elif transport == "sse":
        server.run(transport="sse", host=host, port=port)
    elif transport == "stdio":
        server.run(transport="stdio")
    else:
        raise SystemExit(f"Unknown transport {transport!r} (stdio, sse, streamable-http).")


def _newest_case() -> str | None:
    from .config import get_settings

    cases = sorted(get_settings().cases_dir.glob("*.json"))
    return str(cases[-1]) if cases else None
