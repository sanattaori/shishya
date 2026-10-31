"""Checks of the life-test models. Run with: .venv/bin/python -m pytest brain/tests -q (needs torch)."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brain import data  # noqa: E402
from brain.models import Brain, BrainConfig, build, n_params  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def _pathway_bdh():
    spec = importlib.util.spec_from_file_location("pathway_bdh", HERE / "third_party" / "bdh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plain_bdh_matches_pathway():
    pw = _pathway_bdh()
    torch.manual_seed(0)
    theirs = pw.BDH(pw.BDHConfig(n_layer=3, n_embd=64, n_head=4, dropout=0.0,
                                 mlp_internal_dim_multiplier=8, vocab_size=256)).eval()
    ours = Brain(BrainConfig(n_layer=3, n_embd=64, n_head=4, mult=8, dropout=0.0, lasting=False)).eval()
    with torch.no_grad():
        ours.decoder.copy_(theirs.decoder)
        ours.encoder.copy_(theirs.encoder)
        ours.encoder_v.copy_(theirs.encoder_v)
        ours.embed.weight.copy_(theirs.embed.weight)
        ours.lm_head.copy_(theirs.lm_head)
        idx = torch.randint(0, 256, (2, 40))
        a, _ = theirs(idx)
        b, _ = ours(idx)
    assert torch.allclose(a, b, atol=1e-5)


def _sequential_delta(M0, k, v_next, beta):
    """The delta rule one step at a time: M ← M + β_t k_t (v_{t+1} − Mᵀ k_t)ᵀ."""
    M = M0.clone()
    for t in range(k.size(2)):
        kt = k[:, :, t]                                        # B, nh, N
        err = v_next[:, :, t] - torch.einsum("bhn,bhnm->bhm", kt, M)
        M = M + beta[:, None, t, None, None] * kt[..., None] * err[:, :, None, :]
    return M


def test_parallel_write_is_the_sequential_delta_rule():
    torch.manual_seed(1)
    cfg = BrainConfig(n_layer=2, n_embd=32, n_head=2, mult=8, mem_dim=8, dropout=0.0)
    brain = Brain(cfg).eval()
    idx = torch.randint(0, 256, (3, 17))
    mem = [torch.randn_like(m) * 0.1 for m in brain.new_memory(3)]
    with torch.no_grad():
        brain(idx, memory=mem, keep=True)
        s = torch.rand(3, 17) * 2
        new = brain.write(mem, s, reward=1.5)
        for level, (k, v) in enumerate(brain.trace):
            rate = torch.sigmoid(brain.mem_rate[level])
            nxt = torch.ones(3, 17)
            nxt[:, -1] = 0
            beta = (rate * s * 1.5 * nxt).clamp(max=cfg.max_beta)
            want = _sequential_delta(mem[level], k[:, :, :-1], v[:, :, 1:], beta[:, :-1])
            assert torch.allclose(new[level], want, atol=1e-4)


def test_written_text_is_read_back():
    """After one write, reading the same keys returns (close to) the stored next-step values."""
    torch.manual_seed(2)
    cfg = BrainConfig(n_layer=1, n_embd=32, n_head=2, mult=64, mem_dim=8, dropout=0.0, max_beta=1.0)
    brain = Brain(cfg).eval()
    with torch.no_grad():
        brain.mem_rate.fill_(20.0)                             # rate ≈ 1: store fully
        idx = torch.randint(0, 256, (1, 12))
        brain(idx, keep=True)
        mem = brain.write(brain.new_memory(1), torch.ones(1, 12))
        k, v = brain.trace[0]
        got = k[:, :, :-1] @ mem[0]                            # what each stored key now reads back
        # the last key was written last with β = 1, so it reads back its next-step value exactly
        assert torch.allclose(got[:, :, -1], v[:, :, -1], atol=1e-3)


def test_presets_are_about_the_same_size():
    for preset in ("tiny", "bee"):
        g, b, p = (n_params(build(k, preset)) for k in ("gpt", "bdh", "brain"))
        assert abs(g - b) / b < 0.1, (preset, g, b)
        assert p > b


def test_fact_names_are_split_and_never_real_words():
    split = data.name_split()
    assert split["train"] and split["dev"] and split["test"]
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["dev"]) & set(split["test"])


def test_life_stream_is_deterministic_and_complete():
    stories = [f"Once upon a time there was a cat number {i}. It liked to play. The end." for i in range(400)]
    a = data.life_stream(stories, data.name_split()["test"], seed=3, sizes=data.SMALL_LIFE)
    b = data.life_stream(stories, data.name_split()["test"], seed=3, sizes=data.SMALL_LIFE)
    assert [s.text for s in a.sessions] == [s.text for s in b.sessions]
    seen = {}
    for s in a.sessions:
        for f in s.facts:
            seen[f] = seen.get(f, 0) + 1
    for f in a.facts["main"]:
        assert seen[f.sentence] == 3
    for f in a.facts["once"] + a.facts["reward"]:
        assert seen[f.sentence] == 1
    assert all(len(s.text.encode()) <= data.SESSION_BYTES for s in a.sessions)


def test_window_write_is_the_sequential_delta_rule():
    """v2: sparse keys, one memory per layer; the write must equal the delta rule step by step."""
    torch.manual_seed(3)
    cfg = BrainConfig(n_layer=2, n_embd=32, n_head=2, mult=8, mem_dim=8, dropout=0.0, key="window",
                      key_size=256, key_window=8, key_active=6)
    brain = Brain(cfg).eval()
    idx = torch.randint(0, 256, (2, 15))
    mem = [torch.randn_like(m) * 0.1 for m in brain.new_memory(2)]
    with torch.no_grad():
        brain(idx, memory=mem, keep=True)
        s = torch.rand(2, 15) * 2
        new = brain.write(mem, s, reward=1.5)
        kidx, kval = brain.trace[0][0]
        dense = torch.zeros(2, 15, cfg.key_size).scatter_(-1, kidx, kval)[:, None]      # B, 1, T, units
        for level, (_, v) in enumerate(brain.trace):
            rate = torch.sigmoid(brain.mem_rate[level])
            nxt = torch.ones(2, 15)
            nxt[:, -1] = 0
            beta = (rate * s * 1.5 * nxt).clamp(max=cfg.max_beta)
            want = _sequential_delta(mem[level], dense[:, :, :-1], v[:, :, 1:], beta[:, :-1])
            assert torch.allclose(new[level], want, atol=1e-4)


def test_window_keys_depend_only_on_the_recent_bytes():
    cfg = BrainConfig(n_layer=1, n_embd=32, n_head=2, mult=8, mem_dim=8, key="window", key_size=512,
                      key_window=8, key_active=10)
    brain = Brain(cfg)
    a = torch.tensor([list(b"xxxxxxxxxxxxZubak's cat is")])
    b = torch.tensor([list(b"yyyyyyyyyyyyZubak's cat is")])
    ka, kb = (brain.window_keys(t) for t in (a, b))
    assert torch.equal(ka[0][0, -1], kb[0][0, -1])               # same last 8 bytes: the same key
    assert not torch.equal(ka[0][0, -10], kb[0][0, -10])         # here the window reaches the different prefix
    texts = (b"the cat of Zubak's", b"the cat of Kipos's")
    z, k = (brain.window_keys(torch.tensor([list(t)]))[0][0, -1] for t in texts)
    assert len(set(z.tolist()) & set(k.tolist())) < 5            # a different name in the window: a different key


def test_v1_checkpoints_still_load_as_v1():
    cfg = {k: v for k, v in BrainConfig().__dict__.items() if not k.startswith("key")}
    assert BrainConfig(**cfg).key == "neurons"


def test_window_keys_are_the_fading_sum():
    """The convolution in window_keys computes Σ_j decay^j · key_byte[byte t-j] ⊙ key_pos[j]."""
    cfg = BrainConfig(n_layer=1, n_embd=32, n_head=2, mult=8, mem_dim=8, key="window", key_size=512,
                      key_window=8, key_decay=0.9, key_active=10)
    brain = Brain(cfg)
    idx = torch.randint(0, 256, (2, 20))
    pre = torch.zeros(2, 20, 512)
    for j in range(8):
        pre[:, j:] += 0.9 ** j * brain.key_byte[idx[:, : 20 - j]] * brain.key_pos[j]
    want = pre.topk(10, dim=-1)
    got_idx, got_val = brain.window_keys(idx)
    assert torch.equal(got_idx.sort(-1).values, want.indices.sort(-1).values)
    v = torch.relu(want.values)
    assert torch.allclose(got_val.sort(-1).values, (v / v.norm(dim=-1, keepdim=True)).sort(-1).values, atol=1e-5)
