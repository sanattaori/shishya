"""The three models of the life test, all reading bytes.

- ``GPT``: a plain transformer, the baseline.
- ``Brain`` with ``lasting=False``: plain BDH. Same maths as Pathway's BDH-GPU (``third_party/bdh.py``);
  ``tests/test_brain.py`` checks that both give the same logits.
- ``Brain`` with ``lasting=True``: BDH plus lasting synapses in every layer. They are read on every step and
  written after every text by a local rule, with no gradients (see ``Brain.write``):

      Δ M[i, j] = β_t · k_t[i] · (v_{t+1}[j] − (k_t M)[j])      β_t = min(rate · surprise_t · reward, 1)

  k_t is the unit-length vector of BDH's sparse neurons at step t, v_{t+1} is what came at the next step. This is
  the delta rule, computed exactly in parallel over a text (Yang et al., 2024).

The memory is a list with one tensor per layer, each [batch, heads, neurons, mem_dim]. ``Brain.new_memory``
makes an empty one. Nothing inside the model changes during life: the memory is an explicit state, so it can be
saved, copied and measured.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BrainConfig:
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mult: int = 64                 # neurons per head = mult * n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 256
    lasting: bool = True
    mem_dim: int = 64
    max_surprise: float = 4.0      # surprise weights are capped here
    max_beta: float = 1.0          # a single step never writes more than the full value (keeps the rule stable)
    # where the lasting memory is addressed. "neurons" (v1): BDH's ReLU neurons. "window" (v2): a fixed sparse
    # code of the last key_window bytes, each byte bound to how many steps ago it was (random ±1 codes multiplied
    # together, fading by key_decay per step), keeping the key_active strongest of key_size units.
    # "words" (studied for v3): the same kind of code, but over the last key_window words instead of bytes, so a
    # name four words back counts as much as the word just read. See ``Brain.word_keys``.
    key: str = "neurons"
    key_size: int = 16384
    key_window: int = 32
    key_decay: float = 0.9
    key_active: int = 40
    key_sentence: bool = True      # "words" only: the address starts at the sentence, like a question would
    key_novelty: bool = False      # "words" only: earlier words count by how new they are (see Brain.novelty)
    # write gate (studied for v3): a step is written only if its surprise is at least this. 0 writes every step.
    write_min_surprise: float = 0.0
    # metaplasticity (studied for v3, window and words keys only): a unit's plasticity falls as 1 / (1 + use/tau).
    # 0 turns it off.
    meta_tau: float = 0.0
    # reconsolidation (with meta_tau): a correction (reward > 1) writes with full plasticity, so a protected memory
    # can still be changed when you correct it
    meta_reopen: bool = False

    @property
    def neurons(self) -> int:
        return self.mult * self.n_embd // self.n_head

    @property
    def sparse_key(self) -> bool:
        """True when the lasting memory is addressed by a fixed sparse code (v2 and later), not by the neurons."""
        return self.lasting and self.key in ("window", "words")


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_embd: int = 416
    n_head: int = 8
    dropout: float = 0.1
    vocab_size: int = 256
    block_size: int = 256


PRESETS = {
    "tiny": (BrainConfig(n_layer=4, n_embd=128, n_head=4, mult=16, mem_dim=32),
             GPTConfig(n_layer=4, n_embd=128, n_head=4, block_size=256)),
    "bee": (BrainConfig(n_layer=6, n_embd=256, n_head=4, mult=64, mem_dim=64),
            GPTConfig(n_layer=6, n_embd=416, n_head=8, block_size=256)),
}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _ln(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),), eps=eps)


def _freqs(n: int, theta: float) -> torch.Tensor:
    """BDH's RoPE frequencies (pairs of neurons share one), as in third_party/bdh.py."""
    q = (torch.arange(0, n, 1, dtype=torch.float32) / 2).floor() * 2
    return 1.0 / (theta ** (q / n)) / (2 * math.pi)


def _rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    phases = (phases % 1) * (2 * math.pi)
    return (v * torch.cos(phases)).to(v.dtype) + (v_rot * torch.sin(phases)).to(v.dtype)


MAX_WORD = 16                      # bytes of a word that get their own place code; later bytes share the last
NOVELTY_BUCKETS = 1 << 20          # word-count hash buckets for key_novelty


class Brain(nn.Module):
    """BDH-GPU, with optional lasting synapses."""

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.cfg = cfg
        nh, D, N = cfg.n_head, cfg.n_embd, cfg.neurons
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.embed = nn.Embedding(cfg.vocab_size, D)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.lm_head = nn.Parameter(torch.zeros((D, cfg.vocab_size)).normal_(std=0.02))
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("freqs", _freqs(N, 2**16).view(1, 1, 1, N), persistent=False)
        if cfg.lasting:
            m, L = cfg.mem_dim, cfg.n_layer
            if cfg.sparse_key:
                self.window_code()
            mh = 1 if cfg.sparse_key else nh
            self.mem_in = nn.Parameter(torch.zeros((mh, D, m)).normal_(std=0.02))
            self.mem_out = nn.Parameter(torch.zeros((nh, m, D)).normal_(std=1 / math.sqrt(m)))
            self.mem_gain = nn.Parameter(torch.ones(L))
            self.mem_rate = nn.Parameter(torch.full((L,), -1.0))     # rate = sigmoid(.): about 0.27 at birth
            # the average loss in nats, learned in childhood; surprise = loss / this
            self.register_buffer("loss_ref", torch.tensor(1.0))
        self.life_rate = 1.0                                          # write-rate multiplier, chosen on the dev stream

    def window_code(self):
        """(Re)make v2's fixed random code from the config: the same seed, so the same code in every run."""
        c = self.cfg
        g = torch.Generator().manual_seed(1234)
        rand = lambda n: torch.randint(0, 2, (n, c.key_size), generator=g).float() * 2 - 1  # noqa: E731
        dev = self.lm_head.device
        self.register_buffer("key_byte", rand(c.vocab_size).to(dev))
        self.register_buffer("key_pos", rand(c.key_window).to(dev))
        if c.key == "words":                                  # where a byte sits in its word; "just after a space"
            self.register_buffer("key_inword", rand(MAX_WORD).to(dev), persistent=False)
            self.register_buffer("key_space", rand(1)[0].to(dev), persistent=False)
            if c.key_novelty and not hasattr(self, "key_count"):     # word counts, learned once from childhood text
                self.register_buffer("key_count", torch.zeros(NOVELTY_BUCKETS, device=dev))

    # -- memory ---------------------------------------------------------------------------------------------------
    def new_memory(self, batch: int = 1, device=None) -> list[torch.Tensor]:
        c = self.cfg
        dev = device or self.lm_head.device
        if c.sparse_key:
            mem = [torch.zeros(batch, 1, c.key_size, c.mem_dim, device=dev) for _ in range(c.n_layer)]
            if c.meta_tau > 0:                                    # how much has been written through each unit
                mem.append(torch.zeros(batch, c.key_size, device=dev))
            return mem
        return [torch.zeros(batch, c.n_head, c.neurons, c.mem_dim, device=dev) for _ in range(c.n_layer)]

    @torch.no_grad()
    def window_keys(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """v2 keys: (indices [B, T, active], values [B, T, active]), unit length. Fixed, never trained."""
        c = self.cfg
        if c.key == "words":
            pre = self.word_code(idx)
        else:
            W = c.key_window
            # pre[t] = Σ_j decay^j · key_byte[byte t-j] ⊙ key_pos[j], as one causal depthwise convolution over time
            fade = c.key_decay ** torch.arange(W, device=idx.device, dtype=torch.float32)
            kernel = (self.key_pos * fade[:, None]).flip(0).T.unsqueeze(1).contiguous()     # units, 1, W
            x = self.key_byte[idx].transpose(1, 2)                                          # B, units, T
            pre = F.conv1d(F.pad(x, (W - 1, 0)), kernel, groups=c.key_size).transpose(1, 2)  # B, T, units
        top = pre.topk(c.key_active, dim=-1)
        val = F.relu(top.values)
        return top.indices, val / (val.norm(dim=-1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def word_code(self, idx: torch.Tensor) -> torch.Tensor:
        """The "words" address before the top-k: [B, T, units]. Fixed, never trained.

        A word is a run of letters; any other byte except a space is a word of its own ("Zubak's" is Zubak, ', s).
        A word's code is the sum of its bytes' codes, each bound to its place in the word, divided by √length.
        At step t the address is the word being read so far (or the "just after a space" code), bound to slot 0,
        plus the key_window − 1 words before it, each bound to its slot and fading by key_decay per word. With
        ``key_sentence`` the words stop at the last . ! ? or new line: a fact inside a story and the same words
        asked on their own then get the same address. With ``key_novelty`` each earlier word also counts by how
        new it is (see ``learn_familiarity``): "is" and "called" count little, a name never seen counts fully.

            pre[t] = cur(t) ⊙ slot[0] + Σ_d decay^d · novelty(word) · word(t, d) ⊙ slot[d]
        """
        c = self.cfg
        B, T = idx.shape
        U = c.key_size
        space, n, first, pos = _words(idx)
        e = self.key_byte[idx] * self.key_inword[pos] * (~space)[..., None]             # B, T, U
        # whole words: C[b, j] = Σ codes of word j's bytes / √length (row 0: no word)
        rows = (n + (T + 1) * torch.arange(B, device=idx.device)[:, None]).flatten()
        C = torch.zeros(B * (T + 1), U, device=idx.device).index_add_(0, rows, e.view(B * T, U))
        length = torch.zeros(B * (T + 1), device=idx.device).index_add_(0, rows, (~space).float().flatten())
        scale = length.clamp_min(1).rsqrt()
        if c.key_novelty:
            scale = scale * self.novelty(_word_hash(idx, space, n, pos)).flatten()
        C = (C * scale[:, None]).view(B, T + 1, U)
        # the word being read, so far
        cs = F.pad(e.cumsum(1), (0, 0, 1, 0))                                           # B, T+1, U
        so_far = cs[:, 1:] - cs.gather(1, first[..., None].expand(-1, -1, U))
        so_far = so_far / (pos + 1).float().sqrt()[..., None]
        cur = torch.where(space[..., None], self.key_space.expand(B, T, U), so_far)
        pre = cur * self.key_pos[0]
        sp = space.long()
        if c.key_sentence:                  # words before the last . ! ? or new line are not part of the address
            stop = (idx == 46) | (idx == 33) | (idx == 63) | (idx == 10)
            last_stop = torch.where(stop, n, torch.zeros_like(n)).cummax(1).values
        for d in range(1, c.key_window):
            j = n - d + sp
            j = torch.where(j > last_stop, j, torch.zeros_like(j)) if c.key_sentence else j.clamp_min(0)  # 0: none
            word = C.view(B * (T + 1), U)[(j + (T + 1) * torch.arange(B, device=idx.device)[:, None]).flatten()]
            pre = pre + c.key_decay ** d * word.view(B, T, U) * self.key_pos[d]
        return pre

    def novelty(self, h: torch.Tensor) -> torch.Tensor:
        """How new a word is, from 0 (the most common) to 1 (never seen in childhood): log((1+N)/(1+count))
        divided by log(1+N), where N is the number of words counted."""
        total = self.key_count.sum()
        return torch.log((1 + total) / (1 + self.key_count[h])) / torch.log(1 + total).clamp_min(1e-6)

    @torch.no_grad()
    def learn_familiarity(self, texts: list[str], chunk: int = 256) -> None:
        """Count how often each word occurs in ``texts`` (hashed, like a count-min sketch with one row). Done once
        from childhood text, then fixed. Made-up names never occur there, so they count as new."""
        raw = "\n".join(texts).encode("utf-8")
        n_rows = len(raw) // chunk
        idx = torch.frombuffer(bytearray(raw[: n_rows * chunk]), dtype=torch.uint8).long().view(n_rows, chunk)
        counts = torch.zeros_like(self.key_count)
        for i in range(0, n_rows, 4096):
            part = idx[i:i + 4096].to(counts.device)
            space, n, _, pos = _words(part)
            h = _word_hash(part, space, n, pos)                                      # B, T+1 (row 0: no word)
            counts.index_add_(0, h[:, 1:].flatten(), (h[:, 1:] >= 0).float().flatten() *
                              (torch.arange(1, part.size(1) + 1, device=counts.device) <= n[:, -1:]).flatten())
        self.key_count.copy_(counts)

    # -- forward --------------------------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, memory: list | None = None,
                mask: torch.Tensor | None = None, keep: bool = False):
        """Logits for every position; with ``targets``, the mean loss over positions where ``mask`` is true.

        With ``keep=True`` the keys and next-step values of every layer are kept in ``self.trace`` so that
        ``write`` can store this text afterwards.
        """
        c = self.cfg
        B, T = idx.size()
        D, nh, N = c.n_embd, c.n_head, c.neurons
        x = _ln(self.embed(idx).unsqueeze(1))                                 # B, 1, T, D
        phases = torch.arange(T, device=idx.device, dtype=torch.float32).view(1, 1, -1, 1) * self.freqs
        use_mem = c.lasting and memory is not None
        window = c.sparse_key
        wk = self.window_keys(idx) if window and (use_mem or keep) else None
        trace = []
        for level in range(c.n_layer):
            x_sparse = F.relu(x @ self.encoder)                               # B, nh, T, N
            qr = _rope(phases, x_sparse)
            # fast synapses: this text only. Scaling by 1/N keeps fp16 from overflowing. LayerNorm(c·y, eps·c²)
            # equals LayerNorm(y, eps) exactly, so this is still BDH (checked against third_party/bdh.py).
            yKV = ((qr @ qr.mT) * (1.0 / N)).tril(diagonal=-1) @ x
            yKV = _ln(yKV, eps=1e-5 / N**2)
            if (use_mem or keep) and not window:
                k = x_sparse / (x_sparse.norm(dim=-1, keepdim=True) + 1e-6)
            if use_mem:
                with torch.autocast(device_type=idx.device.type, enabled=False):
                    if window:
                        read = _sparse_read(memory[level], *wk)               # B, 1, T, m
                    else:
                        read = k.float() @ memory[level]                      # B, nh, T, m (always fp32)
                read = read.to(yKV.dtype) @ self.mem_out                      # B, nh, T, D
                yKV = _ln(yKV + self.mem_gain[level] * read)
            if keep:
                v = _ln(x @ self.mem_in)                                       # B, nh or 1, T, m
                trace.append((wk if window else k, v))
            y_sparse = F.relu(yKV @ self.encoder_v)
            xy_sparse = self.drop(x_sparse * y_sparse)                        # B, nh, T, N
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            x = _ln(x + _ln(yMLP))
        logits = x.view(B, T, D) @ self.lm_head
        self.trace = trace if keep else None
        loss = None
        if targets is not None:
            loss = masked_loss(logits, targets, mask)
        return logits, loss

    # -- the local rule -------------------------------------------------------------------------------------------
    def surprise(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Per step: this step's loss divided by the average loss, capped. Steps without a target get 0."""
        with torch.no_grad():
            nll = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
            s = (nll.view_as(targets) / self.loss_ref.clamp_min(1e-3)).clamp(0, self.cfg.max_surprise)
            if mask is not None:
                s = s * mask
        return s

    def write(self, memory: list, surprise: torch.Tensor, reward: torch.Tensor | float = 1.0,
              mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Store the text of the last ``forward(..., keep=True)`` in the lasting synapses; returns the new memory.

        Step t links its key k_t to the value at step t+1. β_t = min(rate · life_rate · surprise_t · reward,
        max_beta). The exact sequential delta rule over the text, computed with one triangular solve:
        E = (I + tril(diag(β) K Kᵀ, -1))⁻¹ (V⁺ − K M₀), then M = M₀ + Kᵀ diag(β) E.
        Gradients flow through it in childhood (the rule's rate and the read path are learned); in life it runs
        under ``torch.no_grad()``.
        """
        c = self.cfg
        assert self.trace is not None, "call forward(..., keep=True) first"
        B = surprise.shape[0]
        if not torch.is_tensor(reward):
            reward = torch.full((B,), float(reward), device=surprise.device)
        valid = torch.ones_like(surprise) if mask is None else mask.float()
        nxt = torch.zeros_like(valid)
        nxt[:, :-1] = valid[:, 1:]                                            # step t needs a real step t+1
        out = []
        if c.write_min_surprise > 0:                                         # the write gate: surprising steps only
            valid = valid * (surprise >= c.write_min_surprise).float()
        if c.sparse_key:
            return self._write_window(memory, surprise, reward, valid, nxt)
        with torch.autocast(device_type=surprise.device.type, enabled=False):
            for level, (k, v) in enumerate(self.trace):
                rate = torch.sigmoid(self.mem_rate[level]) * self.life_rate
                beta = (rate * surprise * reward[:, None] * valid * nxt).clamp(max=c.max_beta)   # B, T
                k = k[:, :, :-1].float()                                      # B, nh, T-1, N
                v_next = v[:, :, 1:].float()                                  # B, nh, T-1, m
                b = beta[:, None, :-1, None]                                  # B, 1, T-1, 1
                M0 = memory[level]
                err0 = v_next - k @ M0
                # A[t, s] = β_s k_t·k_s for s < t. The solver takes the diagonal as 1 and ignores the rest above.
                A = torch.tril(k @ (b * k).mT, diagonal=-1)
                E = torch.linalg.solve_triangular(A, err0, upper=False, unitriangular=True)
                out.append(M0 + k.mT @ (b * E))
        return out

    @torch.no_grad()
    def forget(self, memory: list, mask: torch.Tensor | None = None, passes: int = 8) -> list[torch.Tensor]:
        """Erase the text of the last ``forward(..., keep=True)`` from the lasting synapses; returns the new memory.

        The same delta rule, with 0 as the value to store and full strength at every step: afterwards the text's
        addresses read back (close to) nothing, so the model answers as if it had never read it. This is Larimar's
        forgetting (a write with α = −1), done with the local rule. One pass leaves a little behind where the
        text's addresses overlap; each further pass removes most of the rest. Other memories that share those
        addresses lose a little too.
        """
        assert self.trace is not None, "call forward(..., keep=True) first"
        trace = [(k, torch.zeros_like(v)) for k, v in self.trace]
        k0 = trace[0][0]
        B, T = (k0[0] if isinstance(k0, tuple) else k0[:, 0, :, 0]).shape[:2]
        full = torch.full((B, T), 1e6, device=self.lm_head.device)           # β = rate · 1e6, capped at max_beta
        gate, life = self.cfg.write_min_surprise, self.life_rate
        self.cfg.write_min_surprise, self.life_rate = 0.0, 1.0
        try:
            for _ in range(passes):
                self.trace = trace
                memory = self.write(memory, full, 1.0, mask)
        finally:
            self.cfg.write_min_surprise, self.life_rate = gate, life
            self.trace = None
        return memory

    def _write_window(self, memory, surprise, reward, valid, nxt) -> list[torch.Tensor]:
        """The same delta rule with v2's sparse keys: the Gram matrix once, then per layer a solve and a
        scatter-add into the rows of the active units.

        With ``meta_tau`` > 0 (metaplasticity, studied for v3) each unit i writes with plasticity
        ρ_i = 1 / (1 + use_i / meta_tau), where use_i is how much has been written through it so far
        (Σ β k_i², over its life). The sequential rule becomes M ← M + β (ρ ⊙ k) (v − Mᵀk)ᵀ: a well-used synapse
        changes less, so what it holds lasts. A local, diagonal form of Larimar's least-squares write. ``use``
        is kept as one more tensor at the end of the memory list, shared by the layers. With ``meta_reopen`` a
        correction (reward > 1) writes with ρ = 1 (reconsolidation): protected memories can still be corrected.
        """
        c = self.cfg
        B = surprise.shape[0]
        L = c.n_layer
        out = []
        with torch.autocast(device_type=surprise.device.type, enabled=False):
            kidx, kval = self.trace[0][0]
            kidx, kval = kidx[:, :-1], kval[:, :-1].float()                   # B, T-1, active
            T1 = kidx.size(1)
            dense = torch.zeros(B, T1, c.key_size, device=kidx.device).scatter_(-1, kidx, kval)
            if c.meta_tau > 0:
                use = memory[L]
                if use.size(0) != B:
                    use = use.expand(B, -1)
                rho = 1.0 / (1.0 + torch.gather(use, 1, kidx.reshape(B, -1)).view_as(kval) / c.meta_tau)
                if c.meta_reopen:                                             # reconsolidation: corrections reopen
                    rho = torch.where(reward.view(B, 1, 1) > 1, torch.ones_like(rho), rho)
                kw = kval * rho                                               # the key a write goes through
                G = dense @ torch.zeros_like(dense).scatter_(-1, kidx, kw).mT  # G[t, s] = k_t·(ρ ⊙ k_s)
            else:
                kw = kval
                G = dense @ dense.mT                                          # B, T-1, T-1
            del dense
            rows = (kidx + (torch.arange(B, device=kidx.device) * c.key_size).view(B, 1, 1)).reshape(-1)
            beta_sum = 0
            for level, (_, v) in enumerate(self.trace):
                rate = torch.sigmoid(self.mem_rate[level]) * self.life_rate
                beta = (rate * surprise * reward[:, None] * valid * nxt).clamp(max=c.max_beta)[:, :-1]
                beta_sum = beta_sum + beta
                M0 = memory[level]
                if M0.size(0) != B:
                    M0 = M0.expand(B, -1, -1, -1)
                err0 = v[:, 0, 1:].float() - _sparse_read(M0, kidx, kval)[:, 0]   # B, T-1, m
                A = torch.tril(G * beta[:, None, :], diagonal=-1)             # A[t, s] = β_s k_t·(ρ ⊙ k_s)
                E = torch.linalg.solve_triangular(A, err0, upper=False, unitriangular=True)
                upd = kw[..., None] * (beta[..., None] * E)[:, :, None, :]    # B, T-1, active, m
                new = M0.reshape(B * c.key_size, -1).index_add(0, rows, upd.reshape(-1, upd.size(-1)))
                out.append(new.view_as(M0))
            if c.meta_tau > 0:
                used = (beta_sum / L)[..., None] * kval ** 2                  # B, T-1, active
                out.append(use.reshape(-1).index_add(0, rows, used.reshape(-1)).view_as(use))
        return out


def _words(idx: torch.Tensor):
    """Split bytes into words, as ``Brain.word_code`` does: (space [B, T], word number n from 1, where the word
    started, the byte's place in its word)."""
    B, T = idx.shape
    letter = (idx >= 128) | ((idx | 32) >= 97) & ((idx | 32) <= 122)                   # a-z, A-Z, UTF-8 bytes
    space = (idx == 32) | (idx == 9) | (idx == 13)
    prev = F.pad(letter[:, :-1], (1, 0))
    start = (letter & ~prev) | (~letter & ~space)                                       # a word starts here
    n = start.long().cumsum(1)
    ar = torch.arange(T, device=idx.device).expand(B, T)
    first = torch.where(start, ar, torch.zeros_like(ar)).cummax(1).values
    return space, n, first, (ar - first).clamp(max=MAX_WORD - 1)


def _word_hash(idx, space, n, pos) -> torch.Tensor:
    """A hash in [0, NOVELTY_BUCKETS) for each word: [B, T+1], indexed by word number (row 0: no word)."""
    B, T = idx.shape
    mult = torch.tensor([pow(257, i, NOVELTY_BUCKETS) for i in range(MAX_WORD)], device=idx.device)
    term = (idx * mult[pos]) % NOVELTY_BUCKETS * (~space)
    rows = (n + (T + 1) * torch.arange(B, device=idx.device)[:, None]).flatten()
    h = torch.zeros(B * (T + 1), dtype=torch.long, device=idx.device).index_add_(0, rows, term.flatten())
    return (h % NOVELTY_BUCKETS).view(B, T + 1)


def _sparse_read(M: torch.Tensor, kidx: torch.Tensor, kval: torch.Tensor) -> torch.Tensor:
    """k_t · M for sparse keys. M [B or 1, 1, units, m]; kidx, kval [B, T, active] → [B, 1, T, m]."""
    B, T, a = kidx.shape
    Mf = M[:, 0].float()
    if Mf.size(0) != B:
        Mf = Mf.expand(B, -1, -1)
    rows = torch.gather(Mf, 1, kidx.reshape(B, T * a, 1).expand(-1, -1, Mf.size(-1)))
    return (rows.view(B, T, a, -1) * kval[..., None].float()).sum(2).unsqueeze(1)


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    nll = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
    if mask is None:
        return nll.mean()
    m = mask.flatten().float()
    return (nll * m).sum() / m.sum().clamp_min(1)


class _Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd)
        self.n_head = c.n_head
        self.dropout = c.dropout
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    """A plain pre-LayerNorm transformer with learned positions and a tied output layer."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, memory=None, mask=None, keep=False):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.ln_f(x) @ self.tok.weight.T
        loss = masked_loss(logits, targets, mask) if targets is not None else None
        return logits, loss


# v3 (docs/PLAN.md, amendment of 31 Oct 2026): v2 with these settings, chosen on the dev stream
V3 = {"key": "words", "key_window": 8, "key_decay": 1.0, "key_sentence": True, "key_novelty": True}


def build(kind: str, preset: str) -> nn.Module:
    """kind: gpt | bdh | brain (v1) | brain2 (v2: window keys) | brain3 (v3: see V3)."""
    bcfg, gcfg = PRESETS[preset]
    if kind == "gpt":
        return GPT(gcfg)
    if kind == "bdh":
        return Brain(BrainConfig(**{**asdict(bcfg), "lasting": False}))
    if kind == "brain":
        return Brain(bcfg)
    if kind == "brain2":
        return Brain(BrainConfig(**{**asdict(bcfg), "key": "window"}))
    if kind == "brain3":
        return Brain(BrainConfig(**{**asdict(bcfg), **V3}))
    raise ValueError(f"unknown model {kind!r}")


def config_of(model: nn.Module) -> dict:
    return asdict(model.cfg)


def from_config(kind: str, cfg: dict) -> nn.Module:
    return GPT(GPTConfig(**cfg)) if kind == "gpt" else Brain(BrainConfig(**cfg))
