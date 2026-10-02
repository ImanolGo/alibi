"""FastAPI + Jinja + HTMX web UI. No JS build step.

The UI is thin on purpose: every action calls the same ``Game`` methods the CLI
and the MCP server use, then re-renders the board fragment. Game objects are
cached per process and persisted through the ``GameStore``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .case import Case
from .config import get_settings
from .game import ActionResult, Game, GameStore, Responder, SqlGameStore
from .generator import generate_valid_case
from .llm import BudgetExceededError
from .suspect import SuspectTeam

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

ResponderFactory = Callable[[Case, str], Responder]
CaseGenerator = Callable[[], Case]


def _default_responder(case: Case, game_id: str) -> Responder:
    return SuspectTeam(case, game_id=game_id)


def _default_case() -> Case:
    case, violations = generate_valid_case()
    if violations:
        log.warning("generated case kept %d violation(s)", len(violations))
    return case


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

    app = FastAPI(title="Alibi")
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
        path = cases_dir / f"{case_id}.json"
        return Case.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def get_game(game_id: str) -> Game | None:
        if game_id in cache:
            return cache[game_id]
        game = store.load(game_id)
        if game is not None:
            cache[game_id] = game
        return game

    def start_game(case: Case) -> Game:
        game = get_game(case.id)
        if game is None:
            game = Game(case, game_id=case.id, store=store, responder=make_responder(case, case.id))
            cache[case.id] = game
        return game

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

    # -- routes -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", {"cases": list_cases()})

    @app.post("/case/generate")
    def generate_case_route() -> RedirectResponse:
        case = make_case()
        save_case(case)
        start_game(case)
        return RedirectResponse(url=f"/games/{case.id}", status_code=303)

    @app.post("/games/{case_id}")
    def create_game(case_id: str) -> RedirectResponse:
        case = load_case(case_id)
        if case is None:
            return RedirectResponse(url="/", status_code=303)
        start_game(case)
        return RedirectResponse(url=f"/games/{case_id}", status_code=303)

    @app.get("/games/{game_id}", response_class=HTMLResponse)
    def game_page(request: Request, game_id: str) -> Response:
        game = get_game(game_id)
        if game is None:
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "game.html", {"game": game})

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
        return templates.TemplateResponse(request, "reveal.html", {"game": game})

    return app
