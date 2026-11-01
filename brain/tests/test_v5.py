"""Checks of v5's memory (no LLM needed). Run with: .venv/bin/python -m pytest brain/tests -q (needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from brain import v5  # noqa: E402


class _Reader:
    r, m = 4, 6


class _Student:
    """Just the read path of v5.Student, over a small memory."""

    reader = _Reader()
    read = v5.Student.read


def _steps(n=7, U=50, active=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    kidx = torch.stack([torch.randperm(U, generator=g)[:active] for _ in range(n)])
    if n > 3:
        kidx[3] = kidx[1]                                                 # two steps share a who key
    kval = torch.rand(n, active, generator=g)
    kval = kval / kval.norm(dim=-1, keepdim=True)
    what = torch.randn(n, 4, generator=g)
    what = what / what.norm(dim=-1, keepdim=True)
    V = torch.randn(n, 6, generator=g)
    beta = torch.rand(n, generator=g)
    return kidx, kval, what, V, beta


def test_write_equals_sequential_delta_rule():
    kidx, kval, what, V, beta = _steps()
    st = _Student()
    M0 = torch.randn(50, 4, 6) * 0.1
    fast = v5.write(M0.clone(), kidx, kval, what, V, beta, st)
    M = M0.clone()
    for t in range(len(beta)):                                            # one step at a time
        err = V[t] - st.read(kidx[t:t + 1], kval[t:t + 1], what[t:t + 1], M)[0]
        M[kidx[t]] += kval[t][:, None, None] * what[t][None, :, None] * (beta[t] * err)[None, None, :]
    assert torch.allclose(fast, M, atol=1e-5)


def test_full_strength_write_reads_back_exactly():
    kidx, kval, what, V, _ = _steps(n=1)
    st = _Student()
    M = v5.write(torch.zeros(50, 4, 6), kidx, kval, what, V, torch.ones(1), st)
    assert torch.allclose(st.read(kidx, kval, what, M), V, atol=1e-5)


def test_split_words_matches_the_brain():
    from brain.models import _words
    text = b"Zubak's cat, Fugez!  2 x"
    spans = v5.split_words(text)
    assert [text[s:e] for s, e in spans] == [b"Zubak", b"'", b"s", b"cat", b",", b"Fugez", b"!", b"2", b"x"]
    _, n, _, _ = _words(torch.tensor([list(text)]))
    assert int(n[0, -1]) == len(spans)
