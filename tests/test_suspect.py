from __future__ import annotations

import pytest

from alibi.case import Case
from alibi.fake_llm import FakeLLM
from alibi.suspect import SuspectAgent, Thought

SETTING = {"name": "Test Hall", "year": 1926, "tone": "classy cosy mystery"}


class NullStore:
    """A memory store that stores nothing and recalls nothing."""

    def __init__(self) -> None:
        self.added: list[dict] = []

    def add(self, **kwargs) -> None:
        self.added.append(kwargs)

    def nearest(self, **kwargs) -> list:
        return []


def make_agent(case: Case, person_id: str, *, llm: FakeLLM, store=None) -> SuspectAgent:
    return SuspectAgent(
        case,
        person_id,
        store=store or NullStore(),
        llm=llm,
        game_id="test",
        setting=SETTING,
    )


# --- information hiding ----------------------------------------------------
def test_innocent_prompt_hides_other_secrets_and_the_murderer(valid_case: Case) -> None:
    agent = make_agent(valid_case, "lady_blackwood", llm=FakeLLM())
    prompt = agent.system_prompt()

    # Never another suspect's secret.
    assert "skimming the household accounts" not in prompt  # the butler's
    assert "bookmaker in York" not in prompt  # the doctor's
    # An innocent is never told who the murderer is.
    assert "murderer" not in prompt.lower()
    # No raw internal ids leak into the prose.
    assert "butler_hobbs" not in prompt
    # ...but it does know its own secret and its own scenes.
    assert "meet a lover from the village" in prompt  # its own secret
    assert "Lady Beatrice Blackwood" in prompt


def test_murderer_prompt_knows_the_truth(valid_case: Case) -> None:
    agent = make_agent(valid_case, "butler_hobbs", llm=FakeLLM())
    prompt = agent.system_prompt()
    assert "murderer" in prompt.lower()
    assert "Lord Edmund Blackwood" in prompt  # the victim


# --- the graph runs a full turn -------------------------------------------
def test_answer_runs_all_four_nodes(valid_case: Case) -> None:
    llm = FakeLLM(
        structured=[Thought(worry_level=2, strategy="deny everything", facts_to_use=["kitchen"])],
        completions=["I was in the kitchen all evening."],
    )
    store = NullStore()
    # The murderer keeps the private "think" step (the default for the culprit).
    agent = make_agent(valid_case, "butler_hobbs", llm=llm, store=store)

    answer = agent.answer("Where were you at ten?")

    assert answer == "I was in the kitchen all evening."
    assert agent.last_thought is not None
    assert agent.last_thought.worry_level == 2
    assert agent.history[-2]["role"] == "user"
    assert agent.history[-1]["content"] == answer
    assert store.added  # the remember node stored the exchange
    # retrieve(embed) -> think(structured) -> speak(complete) -> remember(embed)
    assert [call["method"] for call in llm.calls] == ["embed", "structured", "complete", "embed"]


def test_innocent_skips_the_private_plan(valid_case: Case) -> None:
    llm = FakeLLM(completions=["I was at the piano."])
    agent = make_agent(valid_case, "lady_blackwood", llm=llm)

    answer = agent.answer("Where were you at ten?")

    assert answer == "I was at the piano."
    # No structured call: retrieve(embed) -> speak(complete) -> remember(embed)
    assert [call["method"] for call in llm.calls] == ["embed", "complete", "embed"]


def test_reasoning_can_be_forced_on_for_an_innocent(valid_case: Case) -> None:
    llm = FakeLLM(structured=[Thought()], completions=["Hm."])
    agent = SuspectAgent(
        valid_case,
        "lady_blackwood",
        store=NullStore(),
        llm=llm,
        setting=SETTING,
        reason=True,
    )
    agent.answer("Where were you?")
    assert "structured" in [call["method"] for call in llm.calls]


# --- confession only with >= 2 real clues ---------------------------------
def test_confession_unlocks_after_two_pointing_clues(valid_case: Case) -> None:
    llm = FakeLLM(structured=[Thought()] * 2, completions=["Indeed.", "Oh dear."])
    agent = make_agent(valid_case, "butler_hobbs", llm=llm)

    assert agent.pointing_clues_shown() == 0
    assert "may confess" not in agent.system_prompt().lower()

    agent.show_clue("clue_ash")  # points at the butler
    assert agent.pointing_clues_shown() == 1
    assert "may confess" not in agent.system_prompt().lower()

    agent.show_clue("clue_cufflink")  # the second real clue
    assert agent.pointing_clues_shown() == 2
    assert "may confess" in agent.system_prompt().lower()


def test_red_herring_clues_do_not_count(valid_case: Case) -> None:
    llm = FakeLLM(structured=[Thought()], completions=["I have no idea."])
    agent = make_agent(valid_case, "butler_hobbs", llm=llm)
    agent.show_clue("clue_letter")  # red herring, points at Lady Blackwood
    assert agent.pointing_clues_shown() == 0


def test_unknown_clue_raises(valid_case: Case) -> None:
    agent = make_agent(valid_case, "butler_hobbs", llm=FakeLLM())
    with pytest.raises(KeyError):
        agent.show_clue("clue_does_not_exist")
