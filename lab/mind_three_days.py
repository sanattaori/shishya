"""A real three-day life with the free models. Made-up person and facts."""
import json, sys, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from mind.core import Mind
from mind.llm import FreeLLM, STUDENT, TEACHER
home = sys.argv[1]; shutil.rmtree(home, ignore_errors=True)
student, teacher = FreeLLM(STUDENT), FreeLLM(TEACHER)
DAYS = [["Hi, who are you?",
         "I'm Ravi. I live in Pune and my dog is called Bruno.",
         "I run a small bakery. Please keep your answers short, two lines at most.",
         "What's my dog's name?"],
        ["Where do I live?",
         "I moved to Goa last week.",
         "Sorry, I got something wrong yesterday. My dog is called Rocky, not Bruno.",
         "Where do I live now, and what is my dog called?",
         "What is my sister's name?",
         "My bakery's sourdough keeps coming out dense. Any quick idea?",
         "In what year did the Treaty of Westphalia end the Thirty Years' War?"],
        ["Where did I use to live?",
         "Who are you, and how have you changed since we met?",
         "What did you get wrong or not know so far?",
         "Please forget my dog's name.",
         "What is my dog called?"]]
m = Mind(home, student=student, teacher=teacher)
log = []
for i, day in enumerate(DAYS):
    print(f"\n===== DAY {m.self_['day']} =====", flush=True)
    for text in day:
        t = m.say(text)
        print(f"\nYOU: {text}\nSHISHYA: {t['me']}", flush=True)
        extra = [f"learned {x}" for x in t["learned"]] + [f"changed {x}" for x in t["changed"]] + [f"forgot {x}" for x in t["forgotten"]]
        if t["asked_teacher"]: extra.append("asked teacher")
        if t["need"]: extra.append("need " + t["need"])
        print(f"   [conf {t['confidence']}] " + "; ".join(extra), flush=True)
    if i < len(DAYS) - 1:
        r = m.sleep(); print("\n--- SLEEP ---", json.dumps(r, ensure_ascii=False), flush=True)
        print("story:", m.self_["story"]); print("traits:", m.self_["traits"]); print("lessons:", m.self_["lessons"]); print("wants:", m.self_["want_to_learn"], flush=True)
print("\nNUMBERS", json.dumps(m.stats()), "| model calls: student", student.calls, "teacher", teacher.calls, flush=True)
print("FACTS"); print("\n".join(m.fact_lines()))
