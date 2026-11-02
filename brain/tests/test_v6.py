"""v6's childhood worlds: changes come after all first lessons, and the questions ask the newest value."""
import random

import pytest

pytest.importorskip("torch")

from brain import v5, v6  # noqa: E402

T = {r: {"lesson": [f"{{s}} and the {r} {{o}}."], "change": [f"{{s}} has a new {r}, {{o}}."],
         "clause": [f"has a {r} called {{o}}"]} for r in v5.RELATIONS}


def test_changes_follow_lessons_and_questions_ask_the_newest():
    names = [f"N{i}" for i in range(2000)]
    lessons, questions = v6.episode6(random.Random(0), names, v5.TRAIN_RELATIONS, T, p_change=1.0, max_changes=1)
    first_change = next(i for i, x in enumerate(lessons) if " has a new " in x)
    assert first_change == len(lessons) - len(questions)            # every fact changed once, after all lessons
    assert all(" has a new " in x for x in lessons[first_change:])
    for _, a in questions:                                           # the answer is the new value, never the first
        where = [i for i, x in enumerate(lessons) if f" {a}." in x or f" {a}," in x or f" {a} and" in x]
        assert where and all(i >= first_change for i in where)


def test_dense_people_get_one_sentence():
    names = [f"N{i}" for i in range(2000)]
    lessons, questions = v6.episode6(random.Random(1), names, v5.TRAIN_RELATIONS, T, p_dense=1.0, p_change=0.0)
    assert len(lessons) == 8 and len(questions) == 32
    assert all(x.count(" called ") == 4 for x in lessons)
