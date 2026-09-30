"""Typer CLI for Alibi."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .case import Case
from .config import get_settings, load_setting
from .generator import generate_valid_case
from .llm import complete_with_usage, spend_today
from .memory import SqlMemoryStore
from .suspect import SuspectAgent
from .tracing import init_tracing
from .validator import Violation

app = typer.Typer(
    help="Alibi — a murder mystery where every suspect is an LLM.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    init_tracing()


@app.command()
def hello(
    prompt: str = typer.Argument("Say hello as a suspicious butler, in one sentence."),
) -> None:
    """Make one real LLM call and report what it cost."""
    answer, usage = complete_with_usage([{"role": "user", "content": prompt}], role="generator")
    console.print(Panel(answer, title="alibi", border_style="magenta"))
    console.print(
        f"[dim]{usage.model} · {usage.prompt_tokens}+{usage.completion_tokens} tokens · "
        f"${usage.cost_usd:.5f} · {usage.latency_s:.2f}s · today ${spend_today():.4f}[/dim]"
    )


@app.command()
def case(
    seed: int | None = typer.Option(None, "--seed", help="Seed for reproducible generation."),
    out: Path | None = typer.Option(None, "--out", help="Directory for the saved JSON."),
    show_json: bool = typer.Option(False, "--json", help="Print the raw case JSON."),
) -> None:
    """Generate a case, validate it, and save it to cases/."""
    setting = load_setting()
    generated, violations = generate_valid_case(setting=setting, seed=seed)

    if show_json:
        console.print_json(generated.model_dump_json())

    console.print(_render_case(generated, violations))

    destination = (out or get_settings().cases_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{generated.id}.json"
    path.write_text(generated.model_dump_json(indent=2), encoding="utf-8")

    if violations:
        console.print(f"[red]Invalid after repairs[/red] — saved {path} for inspection")
    else:
        console.print(f"[green]Valid case[/green] — saved {path}")
    console.print(f"[dim]spent today: ${spend_today():.4f}[/dim]")

    if violations:
        raise typer.Exit(code=1)


@app.command()
def interrogate(
    case_path: Path = typer.Option(..., "--case", exists=True, dir_okay=False, readable=True),
    suspect: str = typer.Option(..., "--suspect", help="Person id to interrogate."),
    game_id: str = typer.Option("local", "--game-id", help="Groups this game's memories."),
    debug: bool = typer.Option(False, "--debug", help="Show the private 'think' output."),
) -> None:
    """Interrogate one suspect in the terminal. Type /help for commands."""
    case_obj = Case.model_validate_json(case_path.read_text(encoding="utf-8"))
    if case_obj.person(suspect) is None:
        console.print(f"[red]No such suspect: {suspect}[/red]")
        console.print("[dim]Suspects: " + ", ".join(p.id for p in case_obj.suspects()) + "[/dim]")
        raise typer.Exit(code=2)

    store = _make_store()
    agent = SuspectAgent(case_obj, suspect, store=store, game_id=game_id)
    console.print(
        Panel(
            Text.assemble(
                (agent.person.name, "bold magenta"),
                "\n",
                (agent.person.description, "dim"),
                "\n\n",
                "Commands: /clue <id> to show evidence, /debug to toggle thoughts, /quit to leave.",
            ),
            title="interrogation",
            border_style="magenta",
        )
    )

    while True:
        try:
            line = console.input("[bold cyan]You > [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        text = line.strip()
        if not text:
            continue
        if text in {"/quit", "/exit", ":q", "quit", "exit"}:
            break
        if text == "/debug":
            debug = not debug
            console.print(f"[dim]thoughts {'on' if debug else 'off'}[/dim]")
            continue
        if text == "/help":
            console.print("[dim]/clue <id> · /debug · /quit[/dim]")
            continue
        if text.startswith("/clue"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]usage: /clue <clue_id>[/red]")
                continue
            try:
                answer = agent.show_clue(parts[1].strip())
            except KeyError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            _show_turn(agent, answer, debug)
            continue

        _show_turn(agent, agent.answer(text), debug)

    console.print(f"[dim]spent today: ${spend_today():.4f}[/dim]")


def _make_store() -> SqlMemoryStore:
    from .db import get_engine, get_session_factory, init_db

    try:
        init_db(get_engine())
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        console.print(f"[red]Database unavailable:[/red] {exc}")
        console.print("[dim]Run `make up` and check ALIBI_DATABASE_URL in .env.[/dim]")
        raise typer.Exit(code=3) from exc
    return SqlMemoryStore(get_session_factory())


def _show_turn(agent: SuspectAgent, answer: str, debug: bool) -> None:
    console.print(Panel(answer, title=agent.person.name, border_style="green"))
    if debug and agent.last_thought is not None:
        thought = agent.last_thought
        console.print(
            f"[dim]thought: worry {thought.worry_level}/10 · {thought.strategy} · "
            f"{', '.join(thought.facts_to_use) or 'nothing to lean on'}[/dim]"
        )


def _render_case(case_obj, violations: list[Violation]) -> Group:
    persons = Table(title="Persons", show_lines=False)
    persons.add_column("id", style="cyan")
    persons.add_column("name")
    persons.add_column("role")
    persons.add_column("secrets")
    for person in case_obj.persons:
        if person.is_victim:
            role = "[red]victim[/red]"
        elif person.id == case_obj.murderer:
            role = "[yellow]murderer[/yellow]"
        else:
            role = "suspect"
        secrets = "; ".join(
            f"{s.text}{' [red-herring]' if s.is_red_herring else ''}" for s in person.secrets
        )
        persons.add_row(person.id, person.name, role, secrets or "—")

    timeline = Table(title="Timeline", show_lines=False)
    timeline.add_column("time", style="cyan")
    timeline.add_column("location")
    timeline.add_column("actors")
    timeline.add_column("witnesses")
    timeline.add_column("what happens")
    for event in sorted(case_obj.timeline, key=lambda e: e.minutes):
        timeline.add_row(
            event.time,
            event.location,
            ", ".join(event.actors) or "—",
            ", ".join(event.witnesses) or "—",
            event.description,
        )

    clues = Table(title="Clues", show_lines=False)
    clues.add_column("id", style="cyan")
    clues.add_column("location")
    clues.add_column("points to")
    clues.add_column("red herring")
    clues.add_column("description")
    for clue in case_obj.clues:
        clues.add_row(
            clue.id,
            clue.location,
            clue.points_to or "—",
            "yes" if clue.is_red_herring else "no",
            clue.description,
        )

    header = Text.assemble(
        (case_obj.title, "bold magenta"),
        "\n",
        (case_obj.setting, "dim"),
        "\n",
        f"time of death {case_obj.time_of_death} at {case_obj.crime_location} · "
        f"murderer: {case_obj.murderer}",
    )

    parts: list[Any] = [Panel(header, border_style="magenta"), persons, timeline, clues]
    if violations:
        lines = "\n".join(f"[red]{v.code}[/red] {v.message}" for v in violations)
        parts.append(Panel(lines, title="validator violations", border_style="red"))
    else:
        parts.append(Text("validator: no violations ✓", style="green"))
    return Group(*parts)
