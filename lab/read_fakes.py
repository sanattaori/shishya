"""Print the full replies to the made-up questions, to read them by hand."""
import json
import torch
from check_admits_lists import PLAIN_FAKE
from habits import HABITS, Model
from selflayer import LAYER, TEST, SelfLayer, write_with

m = Model(3)
with torch.no_grad():
    size = float(m.net(**m.tok(m.prompt("Why do we dream?"), return_tensors="pt"),
                       output_hidden_states=True).hidden_states[LAYER + 1][0, -1].norm())
layer = SelfLayer(m.net.config.hidden_size, size)
layer.load_state_dict(torch.load("lab/out/self/admits.pt"))
layer.eval()
qs = PLAIN_FAKE + TEST["made_up"]
out = {}
for name, how in (("layer", lambda q: write_with(m, layer, q, 90)), ("usual", lambda q: m.write(q, 90, HABITS["admits"])),
                  ("none", lambda q: m.write(q, 90))):
    with torch.no_grad():
        out[name] = how(qs)
    json.dump({"questions": qs, **out}, open("lab/out/read_fakes.json", "w"), indent=1, ensure_ascii=False)
    print(name, "done", flush=True)
print("READ DONE", flush=True)
