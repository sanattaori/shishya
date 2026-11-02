"""Checks of v7's sleep, erase and clause split (no LLM needed)."""
import random

import pytest

torch = pytest.importorskip("torch")

from brain import v5, v7  # noqa: E402


class _Reader:
    r, m = 4, 6


def _sleeper(U=50):
    s = object.__new__(v7.Sleeper)
    s.reader = _Reader()
    s.M = torch.zeros(U, 4, 6)
    s.K_i, s.K_v, s.Wh, s.tokens, s.beta, s.fact = [], [], [], [], [], []
    s.alive, s.first = {}, {}
    s.replaced = s.sleeps = 0
    s.passes, s.rng, s.resolve = 3, random.Random(0), 0.0
    g = torch.Generator().manual_seed(0)
    z = torch.randn(10, 6, generator=g)
    s.Z = z / z.norm(dim=-1, keepdim=True)
    return s


def _trace(s, kidx, what, tok, fact):
    kval = torch.ones(len(kidx)) / len(kidx) ** 0.5
    s.K_i.append(torch.tensor(kidx, dtype=torch.int16))
    s.K_v.append(kval.half())
    s.Wh.append(what.half())
    s.tokens.append(tok)
    s.beta.append(1.0)
    s.fact.append(fact)
    if fact >= 0 and fact not in s.alive:
        s.alive[fact] = True
        s.first[fact] = len(s.tokens) - 1


def _read(s, i):
    return s.read(s.K_i[i][None].long(), s.K_v[i][None].float(), s.Wh[i][None].float())[0]


def test_clause_starts_split_only_before_small_letters():
    t = "Zubak lives in Kelm, has a cat called Pim, keeps the word Fip and likes Kipos best of all."
    cs = v7.clause_starts(t)
    parts = [t[a:b] for a, b in zip([0] + cs, cs + [len(t)])]
    assert parts == ["Zubak lives in Kelm, ", "has a cat called Pim, ", "keeps the word Fip and ",
                     "likes Kipos best of all."]
    assert v7.clause_starts("Zubak now has a different teacher, Kipos.") == []


def test_sleep_keeps_live_facts_and_drops_dead_ones():
    s = _sleeper()
    w = torch.tensor([1.0, 0, 0, 0])
    _trace(s, [1, 2, 3], w, 4, 0)
    _trace(s, [7, 8, 9], w, 5, 1)
    s.alive[0] = False
    s.sleep()
    assert torch.allclose(_read(s, 1), s.Z[5], atol=1e-2)
    assert _read(s, 0).norm() < 1e-6
    assert s.sleeps == 1


def test_erase_removes_a_written_fact():
    s = _sleeper()
    w = torch.tensor([0, 1.0, 0, 0])
    _trace(s, [1, 2, 3], w, 4, 0)
    s.M = v5.write(s.M, s.K_i[0][None].long(), s.K_v[0][None].float(), s.Wh[0][None].float(), s.Z[4][None],
                   torch.ones(1), s)
    assert torch.allclose(_read(s, 0), s.Z[4], atol=1e-2)
    s._erase(0)
    assert _read(s, 0).norm() < 1e-2
    assert not s.alive[0] and s.replaced == 1


def test_later_wins_at_sleep_only_for_close_addresses():
    s = _sleeper()
    s.resolve = 0.5
    w1, w2 = torch.tensor([1.0, 0, 0, 0]), torch.tensor([0, 1.0, 0, 0])
    _trace(s, [1, 2, 3], w1, 4, 0)          # Zubak lives in Kelm
    _trace(s, [1, 2, 3], w2, 6, 1)          # Zubak has a dog named Rex (same who, other meaning)
    _trace(s, [1, 2, 3], w1, 5, 2)          # Zubak moved to Pell (same who, same meaning, later)
    s._later_wins()
    assert s.alive == {0: False, 1: True, 2: True}
    s.sleep()
    assert torch.allclose(_read(s, 2), s.Z[5], atol=1e-2)
    assert torch.allclose(_read(s, 1), s.Z[6], atol=1e-2)
