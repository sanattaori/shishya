"""The same made-up questions with no help and the usual way, to read by hand next to the layer's replies."""
import json
from check_admits_lists import PLAIN_FAKE
from habits import HABITS, Model
from selflayer import TEST

m = Model(3)
qs = PLAIN_FAKE + TEST["made_up"]
out = {"questions": qs}
out["none"] = m.write(qs, 90)
json.dump(out, open("lab/out/read_fakes2.json", "w"), indent=1, ensure_ascii=False)
print("none done", flush=True)
out["usual"] = m.write(qs, 90, HABITS["admits"])
json.dump(out, open("lab/out/read_fakes2.json", "w"), indent=1, ensure_ascii=False)
print("READ DONE", flush=True)
