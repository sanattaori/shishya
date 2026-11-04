"""Make the research figures (light and dark SVGs).

    python docs/make_research_figures.py     # writes docs/img/{architecture,research,teach}-*.svg

The numbers are the test-set results in docs/RESEARCH.md, written here once so the figures can be rebuilt without
rerunning anything. The animated demo replays docs/demo_output.txt, line by line, sped up. It is pure CSS inside the
SVG. With prefers-reduced-motion it shows the last frame.
"""
from __future__ import annotations

from pathlib import Path

from make_figures import FONT, MONO, THEMES, text

HERE = Path(__file__).resolve().parent
OUT = HERE / "img"

# (question, Shishya, its label, the other, its label)
ROWS = [
    ("Knows a fact after one reading", 96, "Shishya v3", 14, "same-size GPT, fine-tuned"),
    ("Asked in words it never saw", 96.5, "Shishya v5", 96, "RAG on a 135M LLM"),
    ("Five days on: a changed fact, newest value", 74.2, "Shishya v7", 33.3, "RAG-3 on a 135M LLM"),
    ("A change said in other words", 45.6, "Shishya v7", 47.5, "RAG-3 on a 135M LLM"),
    ("Five facts in one sentence", 39, "Shishya v7", 68.5, "RAG-3 on a 135M LLM"),
    ("Two facts combined, no thinking out loud", 29, "Shishya v5", 31, "RAG on a 135M LLM"),
]


def chart(theme: str) -> str:
    t = THEMES[theme]
    W, x0, bar_w, row_h = 900, 24, 560, 92
    H = 92 + row_h * len(ROWS) + 36
    scale = bar_w / 100
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="Shishya against the usual way on six tests. The numbers are in the table below the figure.">',
           f'<rect width="{W}" height="{H}" rx="10" fill="{t["surface"]}"/>',
           text(x0, 34, "What it can do, and what it can't", t["text"], 20, 650),
           text(x0, 56, "Percent right, choosing among 5 names. Test sets only. Guessing scores 20%.", t["text2"], 13)]
    top = 78
    chance = x0 + 20 * scale
    bottom = top + row_h * len(ROWS)
    for f in (0, 50, 100):
        x = x0 + f * scale
        out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{t["line"]}"/>')
        out.append(text(x, bottom + 18, f"{f}%", t["muted"], 12, anchor="middle"))
    for i, (q, a, al, b, bl) in enumerate(ROWS):
        y = top + i * row_h
        out.append(text(x0, y + 20, q, t["text"], 14, 600))
        for j, (v, lab, c) in enumerate(((a, al, t["shishya"]), (b, bl, t["llm"]))):
            yy = y + 28 + j * 22
            w = v * scale
            out.append(f'<path d="M{x0},{yy} H{x0 + w - 4} Q{x0 + w},{yy} {x0 + w},{yy + 4} V{yy + 12} '
                       f'Q{x0 + w},{yy + 16} {x0 + w - 4},{yy + 16} H{x0} Z" fill="{c}"/>')
            lx = max(x0 + w + 8, chance + 8)         # a label never sits on the guessing line
            out.append(text(lx, yy + 13, f"{v:g}%  {lab}", t["text2"], 13, 600 if j == 0 else 400))
    for i in range(len(ROWS)):                       # only across the bars, so it never crosses a title
        y = top + i * row_h
        out.append(f'<line x1="{chance}" y1="{y + 26}" x2="{chance}" y2="{y + 68}" stroke="{t["text2"]}" '
                   f'stroke-dasharray="3 4"/>')
    out.append(text(chance, bottom + 18, "guessing", t["text2"], 12, anchor="middle"))
    out.append("</svg>")
    return "".join(out)


def architecture(theme: str) -> str:
    t = THEMES[theme]
    W, H = 900, 500
    blue = t["shishya"]

    def box(x, y, w, h, title, lines, fill=None, ink=None, ink2=None):
        fill, ink, ink2 = fill or t["card"], ink or t["text"], ink2 or t["text2"]
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{t["line"]}"/>',
               text(x + 14, y + 24, title, ink, 14, 650)]
        out += [text(x + 14, y + 46 + i * 19, ln, ink2, 12.5) for i, ln in enumerate(lines)]
        return "".join(out)

    def arrow(points, label=None, at=None):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        (x0, y0), (x1, y1) = points[-2], points[-1]
        ux, uy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
        head = (f"{x1},{y1} {x1 - 8 * ux + 4 * uy},{y1 - 8 * uy + 4 * ux} "
                f"{x1 - 8 * ux - 4 * uy},{y1 - 8 * uy - 4 * ux}")
        s = (f'<path d="{d}" fill="none" stroke="{t["text2"]}" stroke-width="1.5"/>'
             f'<polygon points="{head}" fill="{t["text2"]}"/>')
        if label:
            s += text(at[0], at[1], label, t["text2"], 12)
        return s

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
           'aria-label="Architecture. Text goes to a frozen BDH reader and to the who part of the address. The reader '
           'gives the what part. Who and what address a lasting memory, the only part that changes. The memory nudges '
           'the next letter.">',
           f'<rect width="{W}" height="{H}" rx="10" fill="{t["surface"]}"/>',
           text(24, 34, "Inside Shishya", t["text"], 20, 650),
           text(24, 56, "Only the blue part changes after training.", t["text2"], 13),
           box(24, 92, 220, 76, "The text", ["Priya lives in a town called"]),
           box(300, 76, 340, 164, "Reader: BDH  (100.9M, frozen)",
               ["1  Neurons fire", "2  They link to earlier bytes", "3  Neurons fire again",
                "4  A neuron stays on only if both agree", "5  Update, then repeat: 6 rounds"]),
           box(696, 92, 180, 76, "Next letter", ["N", "guess + nudge"]),
           box(24, 288, 220, 92, "Who  (frozen)", ["The rare name picks", "40 of 16,384 rows"]),
           box(300, 288, 220, 92, "What  (frozen)", ["What kind of fact this is:", "32 numbers"]),
           box(580, 288, 296, 180, "Lasting memory", ["16,384 × 32 × 128 numbers", "The only part that changes", "",
                                                      "To learn, it writes the next letter:",
                                                      "M += β · who · what ⊗ error",
                                                      "No gradients. No stored text."],
               fill=blue, ink="#ffffff", ink2="#ffffff"),
           arrow([(244, 130), (300, 130)]),
           arrow([(640, 130), (696, 130)]),
           arrow([(134, 168), (134, 288)]),
           arrow([(410, 240), (410, 288)], "reader's state", (418, 268)),
           arrow([(520, 334), (580, 334)]),
           arrow([(134, 380), (134, 432), (580, 432)], "picks the row", (300, 424)),
           arrow([(786, 288), (786, 168)], "nudges", (794, 232)),
           "</svg>"]
    return "".join(out)


def demo(theme: str) -> str:
    t = THEMES[theme]
    lines = (HERE / "demo_output.txt").read_text().splitlines()
    W, row, head, pad = 900, 20, 42, 22
    H = head + 18 + row * len(lines) + 16
    total = 0.5 * len(lines) + 5.0                  # seconds per loop: then a 5 s hold
    css = [f".a{{animation-duration:{total}s;animation-iteration-count:infinite;animation-timing-function:linear}}"
           "@media (prefers-reduced-motion:reduce){.a{animation:none!important}}"]
    body = []
    y = head + 26
    for i, line in enumerate(lines):
        at = 0.5 * i
        p0, p1 = at / total * 100, (at + 0.12) / total * 100
        css.append(f"@keyframes k{i}{{0%,{p0:.3f}%{{opacity:0}}{p1:.3f}%,100%{{opacity:1}}}}"
                   f".k{i}{{animation-name:k{i}}}")
        if line.startswith("#"):
            colour, weight = t["muted"], 400
        elif line.lstrip().startswith("now:"):
            colour, weight = (t["bad"] if "Nashik and is" in line or "Star Wars" in line else t["good"]), 700
        elif line.lstrip().startswith("before reading"):
            colour, weight = t["muted"], 400
        else:
            colour, weight = t["text"], 400
        body.append(f'<g class="a k{i}">{text(pad, y, line, colour, 13.5, weight, family=MONO)}</g>')
        y += row
    head_svg = (f'<rect width="{W}" height="{H}" rx="10" fill="{t["card"]}" stroke="{t["line"]}"/>'
                f'<path d="M0,10 Q0,0 10,0 H{W - 10} Q{W},0 {W},10 V{head} H0 Z" fill="{t["surface"]}"/>'
                f'<line x1="0" y1="{head}" x2="{W}" y2="{head}" stroke="{t["line"]}"/>'
                + text(W / 2, head / 2 + 5, "python -m brain.teach --tour  (sped up)", t["text2"], 14, 600, "middle",
                       family=FONT))
    label = "Terminal replay: Shishya reads three sentences once, then answers from memory, with two failures shown."
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="{label}"><style>{"".join(css)}</style>{head_svg}{"".join(body)}</svg>')


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        (OUT / f"architecture-{theme}.svg").write_text(architecture(theme))
        (OUT / f"research-{theme}.svg").write_text(chart(theme))
        (OUT / f"teach-{theme}.svg").write_text(demo(theme))
