"""Talk to Shishya.

    python -m mind chat                 talk, type /help for the commands
    python -m mind say "hello"          one message
    python -m mind sleep                end the day
    python -m mind self                 who it is now
    python -m mind memory               what it knows
"""
from __future__ import annotations

import argparse
import json

from .core import Mind
from .llm import STUDENT, TEACHER, FreeLLM, LLMError

HELP = "/self  who it is now\n/memory  what it knows\n/sleep  end the day\n/undo  undo the last sleep\n/quit  leave"


def show_turn(mind: Mind, turn: dict) -> None:
    print(f"\n{mind.self_['name']}: {turn['me']}")
    notes = [f"learned {x}" for x in turn["learned"]] + [f"changed {x}" for x in turn["changed"]]
    notes += [f"forgot {x}" for x in turn["forgotten"]]
    if notes:
        print("   (" + "; ".join(notes) + ")")


def show_self(mind: Mind) -> None:
    s = mind.self_
    print(f"{s['name']}, day {s['day']}, born {s['born']}, slept {s['sleeps']} times")
    print("How it is: " + ", ".join(s["traits"]))
    print("Its story: " + s["story"])
    for title, key in (("What it learned about helping", "lessons"), ("What it wants to learn", "want_to_learn")):
        if s[key]:
            print(title + ":")
            for x in s[key]:
                print("  - " + x)
    print("Numbers: " + json.dumps(mind.stats()))


def show_sleep(result: dict) -> None:
    if not result["slept"]:
        print(f"It did not sleep: {result['why']}.")
        return
    print(f"It slept. Now day {result['day']}. {result['what_changed']}")
    if result["traits_before"] != result["traits_now"]:
        print("How it is now: " + ", ".join(result["traits_now"]))


def show_memory(mind: Mind) -> None:
    print("\n".join(mind.fact_lines()) or "It knows nothing yet.")


def chat(mind: Mind) -> None:
    show_self(mind)
    print("\nType to talk. /help for commands.")
    actions = {"/self": show_self, "/memory": show_memory, "/sleep": lambda m: show_sleep(m.sleep()),
               "/undo": lambda m: print("Undone." if m.undo() else "Nothing to undo."),
               "/help": lambda m: print(HELP)}
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "/quit":
            return
        if text in actions:
            actions[text](mind)
        elif text:
            try:
                show_turn(mind, mind.say(text))
            except LLMError as e:
                print(f"(no answer: {e}. Nothing was saved. Try again in a minute.)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mind", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["chat", "say", "sleep", "self", "memory", "undo"])
    p.add_argument("text", nargs="?")
    p.add_argument("--dir", default="~/.shishya/mind", help="the folder that holds its whole life")
    a = p.parse_args(argv)
    mind = Mind(a.dir, student=FreeLLM(STUDENT), teacher=FreeLLM(TEACHER))
    try:
        if a.command == "chat":
            chat(mind)
        elif a.command == "say":
            if not a.text:
                p.error("say needs a message")
            show_turn(mind, mind.say(a.text))
        elif a.command == "sleep":
            show_sleep(mind.sleep())
        elif a.command == "self":
            show_self(mind)
        elif a.command == "memory":
            show_memory(mind)
        else:
            print("Undone." if mind.undo() else "Nothing to undo.")
    except LLMError as e:
        print(f"No answer: {e}. Nothing was saved. Try again in a minute.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
