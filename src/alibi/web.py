"""FastAPI + Jinja + HTMX web UI. No JS build step.

The UI is thin on purpose: every action calls the same ``Game`` methods the CLI
and the MCP server use, then re-renders the board fragment. Game objects are
cached per process and persisted through the ``GameStore``.

Each "Open the case" starts a fresh game with its own id (and its own suspect
memories); the game URL is the save. Only presentation helpers live here.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import escape

from .case import Case, format_time, parse_time
from .config import get_settings
from .game import (
    MAX_ACTIONS,
    MAX_LIE_DETECTOR,
    ActionResult,
    Game,
    GameStore,
    ResponderFactory,
    SqlGameStore,
)
from .generator import generate_valid_case
from .llm import BudgetExceededError, StructuredOutputError
from .suspect import SuspectTeam

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"  # vendored htmx + fonts
LAST_GAME_COOKIE = "alibi_last_game"

CaseGenerator = Callable[[], Case]

# The story clock: you arrive an hour after the death; every action takes a
# quarter of an hour, and the roads clear at dawn (when your actions run out).
MINUTES_PER_ACTION = 15
ARRIVAL_DELAY_MIN = 60


def _default_responder(case: Case, game_id: str) -> SuspectTeam:
    return SuspectTeam(case, game_id=game_id)


def _default_case() -> Case:
    case, violations = generate_valid_case()
    if violations:
        log.warning("generated case kept %d violation(s)", len(violations))
    return case


# -- presentation helpers (pure, unit-tested) -------------------------------
@dataclass(frozen=True)
class Clock:
    now: str
    dawn: str
    used_fraction: float
    urgent: bool


def story_clock(game: Game) -> Clock:
    """Map actions to an evening: arrival → dawn. Display only; rules are unchanged."""
    arrival = parse_time(game.case.time_of_death) + ARRIVAL_DELAY_MIN
    used = MAX_ACTIONS - game.state.actions_left
    day = 24 * 60
    return Clock(
        now=format_time((arrival + used * MINUTES_PER_ACTION) % day),
        dawn=format_time((arrival + MAX_ACTIONS * MINUTES_PER_ACTION) % day),
        used_fraction=used / MAX_ACTIONS,
        urgent=game.state.outcome == "playing" and game.state.actions_left <= 3,
    )


def _room_name(game: Game, location_id: str) -> str:
    room = game.case.location(location_id)
    return room.name if room else location_id


def clue_cards(game: Game) -> list[dict[str, str]]:
    """Discovered clues as the player knows them: description and where found."""
    cards = []
    for clue_id in game.state.discovered_clues:
        clue = game.case.clue(clue_id)
        if clue is not None:
            cards.append(
                {
                    "id": clue.id,
                    "description": clue.description,
                    "room": _room_name(game, clue.location),
                }
            )
    return cards


_CLUE_ID_NOTE = re.compile(r"\s*\(clue id: [^)]*\)")


def friendly(message: str) -> str:
    """Drop the machine ids the MCP/CLI need from text shown to a human."""
    return _CLUE_ID_NOTE.sub("", message)


def detective_rank(game: Game) -> dict[str, str]:
    """A playful verdict on the investigation (reveal page)."""
    used = MAX_ACTIONS - game.state.actions_left
    if game.state.outcome == "won":
        if used <= 10:
            return {
                "rank": "Great Detective",
                "note": f"Solved in {used} moves. The papers will be insufferable about it.",
            }
        if used <= 15:
            return {"rank": "Chief Inspector", "note": f"Solved in {used} moves. Tidy work."}
        return {"rank": "Inspector", "note": "Solved just before dawn. Cutting it fine."}
    if game.state.accused:
        return {
            "rank": "Sergeant",
            "note": "You named someone. Unfortunately, not the right someone.",
        }
    return {
        "rank": "Constable",
        "note": "The roads cleared at dawn, and the murderer left with them.",
    }


def case_review(game: Game) -> dict[str, Any]:
    """What the player found versus what there was to find."""
    murderer = game.case.murderer
    genuine = [c for c in game.case.clues if not c.is_red_herring and c.points_to == murderer]
    found = set(game.state.discovered_clues)
    shown = set(game.state.presented.get(murderer, []))
    return {
        "clues_found": len(found),
        "clues_total": len(game.case.clues),
        "genuine_found": sum(1 for c in genuine if c.id in found),
        "genuine_total": len(genuine),
        "shown_to_murderer": sum(1 for c in genuine if c.id in shown),
        "missed": [
            {"description": c.description, "room": _room_name(game, c.location)}
            for c in genuine
            if c.id not in found
        ],
        "actions_used": MAX_ACTIONS - game.state.actions_left,
    }


def create_app(
    *,
    game_store: GameStore | None = None,
    responder_factory: ResponderFactory | None = None,
    case_dir: Path | None = None,
    case_generator: CaseGenerator | None = None,
) -> FastAPI:
    settings = get_settings()
    store = game_store or SqlGameStore()
    make_responder = responder_factory or _default_responder
    make_case = case_generator or _default_case
    cases_dir = Path(case_dir) if case_dir else settings.cases_dir
    cases_dir.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["friendly"] = friendly
    templates.env.globals.update(
        story_clock=story_clock,
        clue_cards=clue_cards,
        MAX_ACTIONS=MAX_ACTIONS,
        MAX_LIE_DETECTOR=MAX_LIE_DETECTOR,
    )

    app = FastAPI(title="Alibi")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    cache: dict[str, Game] = {}

    # -- helpers ------------------------------------------------------------
    def list_cases() -> list[Case]:
        found: list[Case] = []
        for path in sorted(cases_dir.glob("*.json")):
            try:
                found.append(Case.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - ignore unreadable files
                log.warning("skipping unreadable case file %s", path)
        return found

    def save_case(case: Case) -> None:
        (cases_dir / f"{case.id}.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")

    def load_case(case_id: str) -> Case | None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            return None  # never let a case id walk out of the cases directory
        path = cases_dir / f"{case_id}.json"
        return Case.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def get_game(game_id: str) -> Game | None:
        if game_id in cache:
            return cache[game_id]
        game = store.load(game_id, responder_factory=make_responder)
        if game is not None:
            cache[game_id] = game
        return game

    def start_game(case: Case) -> Game:
        # A fresh game every time: replays work, and visitors of a public
        # deploy no longer share (and spoil) one game per case.
        game_id = f"{case.id}-{uuid.uuid4().hex[:8]}"
        game = Game(case, game_id=game_id, store=store, responder=make_responder(case, game_id))
        cache[game_id] = game
        return game

    def remember_game(response: Response, game: Game) -> None:
        response.set_cookie(
            LAST_GAME_COOKIE, game.game_id, max_age=60 * 60 * 24 * 30, samesite="lax"
        )

    def to_game(game: Game) -> RedirectResponse:
        response = RedirectResponse(url=f"/games/{game.game_id}", status_code=303)
        remember_game(response, game)
        return response

    def board(request: Request, game: Game, result: ActionResult | None = None) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "partials/board.html", {"game": game, "result": result}
        )

    def safe(action: Callable[[], ActionResult]) -> ActionResult:
        try:
            return action()
        except BudgetExceededError:
            return ActionResult(
                ok=False,
                message="The detective is out of budget today — come back tomorrow.",
            )
        except (TimeoutError, StructuredOutputError) as exc:
            log.warning("model trouble during an action: %s", exc)
            return ActionResult(
                ok=False,
                message="The suspect stares at you blankly (the model did not answer). Try again.",
            )

    def index_page(request: Request, error: str | None = None) -> Response:
        last = None
        last_id = request.cookies.get(LAST_GAME_COOKIE)
        if last_id:
            last = get_game(last_id)
        return templates.TemplateResponse(
            request, "index.html", {"cases": list_cases(), "last": last, "error": error}
        )

    # -- routes -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return index_page(request)

    @app.post("/case/generate")
    def generate_case_route(request: Request) -> Response:
        htmx = request.headers.get("HX-Request") == "true"
        try:
            case = make_case()
        except BudgetExceededError:
            message = "The detective is out of budget today. Come back tomorrow."
        except Exception as exc:  # noqa: BLE001 - a failed generation must not 500
            log.warning("case generation failed: %s", exc)
            message = "The writer tore up the draft (generation failed). Try again in a moment."
        else:
            save_case(case)
            game = start_game(case)
            if htmx:
                response = Response(
                    status_code=204, headers={"HX-Redirect": f"/games/{game.game_id}"}
                )
                remember_game(response, game)
                return response
            return to_game(game)
        if htmx:
            return HTMLResponse(f'<p class="note bad" role="alert">{escape(message)}</p>')
        return index_page(request, error=message)

    @app.post("/games/{case_id}")
    def create_game(case_id: str) -> RedirectResponse:
        case = load_case(case_id)
        if case is None:
            return RedirectResponse(url="/", status_code=303)
        return to_game(start_game(case))

    @app.get("/games/{game_id}", response_class=HTMLResponse)
    def game_page(request: Request, game_id: str) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "game.html", {"game": game, "result": None})

    @app.post("/games/{game_id}/search", response_class=HTMLResponse)
    def search(request: Request, game_id: str, room_id: str = Form(...)) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return board(request, game, safe(lambda: game.search_room(room_id)))

    @app.post("/games/{game_id}/inspect", response_class=HTMLResponse)
    def inspect_clue(request: Request, game_id: str, clue_id: str = Form(...)) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return board(request, game, safe(lambda: game.inspect(clue_id)))

    @app.post("/games/{game_id}/question", response_class=HTMLResponse)
    def question(
        request: Request,
        game_id: str,
        suspect_id: str = Form(...),
        text: str = Form(...),
        evidence: list[str] = Form(default=[]),
    ) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return board(request, game, safe(lambda: game.question(suspect_id, text, evidence)))

    @app.post("/games/{game_id}/detector", response_class=HTMLResponse)
    def detector(request: Request, game_id: str, suspect_id: str = Form(default="")) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return board(request, game, safe(lambda: game.lie_detector(suspect_id or None)))

    @app.post("/games/{game_id}/accuse", response_class=HTMLResponse)
    def accuse(
        request: Request,
        game_id: str,
        suspect_id: str = Form(...),
        motive: str = Form(default=""),
    ) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return board(request, game, safe(lambda: game.accuse(suspect_id, motive or None)))

    @app.get("/games/{game_id}/reveal", response_class=HTMLResponse)
    def reveal(request: Request, game_id: str) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        if game.state.outcome == "playing":
            # No peeking: the solution is only shown once the game is over.
            return RedirectResponse(url=f"/games/{game_id}", status_code=303)
        return templates.TemplateResponse(
            request,
            "reveal.html",
            {"game": game, "rank": detective_rank(game), "review": case_review(game)},
        )

    return app
