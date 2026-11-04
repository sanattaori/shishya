"""The mind's own logic, with a scripted model: no network."""
import json
from datetime import datetime, timedelta

import pytest

from mind.core import Mind
from mind.llm import FreeLLM, parse_json


class Script:
    """A model that gives the prepared replies in order and records what it was asked."""

    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def __call__(self, prompt, system=""):
        self.asked.append((prompt, system))
        return self.replies.pop(0)


def said(answer="ok", remember=(), confidence=0.9, need="", forget=()):
    return json.dumps({"answer": answer, "remember": list(remember), "forget": list(forget),
                       "confidence": confidence, "need": need})


def clock():
    t = [datetime(2026, 11, 3, 9, 0).astimezone()]

    def now():
        t[0] += timedelta(hours=1)
        return t[0]
    return now


DREAM = json.dumps({"story": "I met my person.", "traits": ["curious"], "lessons": ["keep it short"],
                    "want_to_learn": ["their sister's name"], "what_changed": "I answer more briefly.",
                    "cards": [{"q": "Where does Ravi live?", "a": "Goa"}, {"q": 3}]})


def test_a_new_value_replaces_the_old_and_keeps_it_as_history(tmp_path):
    home = {"about": "Ravi", "what": "home", "value": "Pune"}
    m = Mind(tmp_path, Script(said(remember=[home]), said(remember=[dict(home, value="Goa")])), clock=clock())
    assert m.say("I live in Pune")["learned"] == ["Ravi, home: Pune"]
    assert m.say("I moved to Goa")["changed"] == ["Ravi, home: Pune is now Goa"]
    assert [x["value"] for x in m.facts] == ["Goa"]
    assert m.facts[0]["old"][0]["value"] == "Pune"
    assert "used to be Pune" in m.fact_lines()[0]


def test_the_same_value_or_other_spacing_is_not_a_change(tmp_path):
    m = Mind(tmp_path, Script(said(remember=[{"about": "Ravi", "what": "home", "value": "Goa"}]),
                              said(remember=[{"about": " ravi", "what": "HOME", "value": "goa"}])), clock=clock())
    m.say("a")
    t = m.say("b")
    assert (t["learned"], t["changed"], len(m.facts)) == ([], [], 1)


def test_memory_and_self_survive_a_restart(tmp_path):
    m = Mind(tmp_path, Script(said(remember=[{"about": "Ravi", "what": "dog", "value": "Bruno"}]), DREAM),
             clock=clock())
    m.say("my dog is Bruno")
    m.sleep()
    again = Mind(tmp_path, Script(said()), clock=clock())
    assert again.facts[0]["value"] == "Bruno"
    assert again.self_["day"] == 2 and again.self_["story"] == "I met my person."
    again.say("hi")
    system = again.student.asked[0][1]
    assert "Bruno" in system and "I met my person." in system and "keep it short" in system


def test_sleep_saves_a_version_and_undo_goes_back(tmp_path):
    m = Mind(tmp_path, Script(said(), DREAM), clock=clock())
    m.say("hi")
    r = m.sleep()
    assert r["slept"] and r["cards"] == 1 and m.self_["traits"] == ["curious"]
    assert m.undo()
    assert m.self_["story"].startswith("I have just started")
    assert m.self_["version"] == 3


def test_a_dream_that_makes_no_sense_changes_nothing(tmp_path):
    for bad in ("not json", json.dumps({"story": "x"}), json.dumps({"story": " ", "traits": ["a"], "lessons": [],
                                                                    "want_to_learn": []})):
        m = Mind(tmp_path / str(hash(bad)), Script(said(), bad, bad), clock=clock())
        m.say("hi")
        before = dict(m.self_)
        assert not m.sleep()["slept"]
        assert m.self_ == before


def test_a_broken_dream_gets_one_more_try(tmp_path):
    m = Mind(tmp_path, Script(said(), "broken {", DREAM), clock=clock())
    m.say("hi")
    assert m.sleep()["slept"]


def test_no_day_no_sleep(tmp_path):
    assert not Mind(tmp_path, Script(), clock=clock()).sleep()["slept"]


def test_it_asks_the_teacher_only_when_unsure_and_a_teacher_could_know(tmp_path):
    teacher = Script("Canberra.")
    m = Mind(tmp_path, Script(said("maybe", confidence=0.2, need="teacher"),
                              said("yes", confidence=0.9, need="teacher"),
                              said("I don't know, what is it?", confidence=0.1, need="person")),
             teacher=teacher, clock=clock())
    t = m.say("capital of Australia?")
    assert t["asked_teacher"] and "Canberra" in t["me"] and "asked my teacher" in t["me"]
    assert not m.say("sure thing?")["asked_teacher"]
    assert not m.say("my sister's name?")["asked_teacher"]
    assert len(teacher.asked) == 1
    assert m.stats()["asked_teacher"] == 1 and m.stats()["answered_alone"] == 2


def test_teacher_answers_are_not_stored_as_facts_the_person_gave(tmp_path):
    m = Mind(tmp_path, Script(said("?", confidence=0.1, need="teacher")), teacher=Script("Canberra."), clock=clock())
    m.say("capital of Australia?")
    assert m.facts == []
    assert any(n["who"] == "teacher" for n in m._rows("notes.jsonl"))


def test_forget_on_request(tmp_path):
    fact = {"about": "Ravi", "what": "dog", "value": "Bruno"}
    m = Mind(tmp_path, Script(said(remember=[fact]), said(forget=[{"about": "ravi", "what": "dog"}])), clock=clock())
    m.say("my dog is Bruno")
    assert m.say("forget my dog")["forgotten"] == ["Ravi, dog"]
    assert m.facts == []
    assert json.loads((tmp_path / "facts.json").read_text()) == []


def test_a_reply_in_the_wrong_shape_is_still_an_answer_and_teaches_nothing(tmp_path):
    m = Mind(tmp_path, Script("Hello there!"), clock=clock())
    t = m.say("hi")
    assert t["me"] == "Hello there!" and t["learned"] == [] and t["confidence"] == 0.0


@pytest.mark.parametrize("bad", [None, "x", [1], [{"about": "a"}], [{"about": 1, "what": "b", "value": "c"}],
                                 [{"about": "a", "what": " ", "value": "c"}]])
def test_broken_memories_are_ignored(tmp_path, bad):
    m = Mind(tmp_path, Script(json.dumps({"answer": "ok", "remember": bad, "forget": bad, "confidence": "high"})),
             clock=clock())
    t = m.say("hi")
    assert m.facts == [] and t["confidence"] == 0.0


def test_recall_finds_older_messages_by_their_words(tmp_path):
    m = Mind(tmp_path, Script(said(), said(), said()), clock=clock())
    m.say("The meeting with Acme is on Friday")
    m.say("I like tea")
    assert [n["text"] for n in m.recall("when is the Acme meeting")] == ["The meeting with Acme is on Friday"]
    assert m.recall("zebra") == []


def test_only_free_models_are_allowed():
    with pytest.raises(ValueError, match="free"):
        FreeLLM(("openai/gpt-5",), key="x")


def test_parse_json_copes_with_fences_and_trailing_commas():
    assert parse_json('Sure!\n```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert parse_json("no json") is None
    assert parse_json("[1, 2]") is None
