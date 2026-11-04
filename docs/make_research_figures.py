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
    ("Two facts combined, no thinking out loud", 27.5, "Shishya v5", 31, "RAG on a 135M LLM"),
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
    W, H = 900, 590
    blue = t["shishya"]

    def stage(x, num, title, lines, active=False):
        fill, ink, ink2 = (blue, "#ffffff", "#ffffff") if active else (t["card"], t["text"], t["text2"])
        out = [f'<rect x="{x}" y="92" width="268" height="132" rx="8" fill="{fill}" stroke="{t["line"]}"/>',
               text(x + 14, 118, f"{num}. {title}", ink, 15, 650)]
        out += [text(x + 14, 142 + i * 19, ln, ink2, 12.5) for i, ln in enumerate(lines)]
        return "".join(out)

    def arrow(x0, y0, x1, y1):
        ux, uy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
        head = (f"{x1},{y1} {x1 - 8 * ux + 4 * uy},{y1 - 8 * uy + 4 * ux} "
                f"{x1 - 8 * ux - 4 * uy},{y1 - 8 * uy - 4 * ux}")
        return (f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{t["text2"]}" stroke-width="1.5"/>'
                f'<polygon points="{head}" fill="{t["text2"]}"/>')

    def panel(x, w, title, lines, fill=None, ink=None, ink2=None):
        fill, ink, ink2 = fill or t["card"], ink or t["text"], ink2 or t["text2"]
        out = [f'<rect x="{x}" y="296" width="{w}" height="270" rx="8" fill="{fill}" stroke="{t["line"]}"/>',
               text(x + 14, 322, title, ink, 15, 650)]
        y0 = 480 if w > 200 else 352
        out += [text(x + 14, y0 + i * 19, ln, ink2, 12.5) for i, ln in enumerate(lines)]
        return "".join(out)

    # neurons: a few fire. The pattern is only a picture.
    lit = {2, 9, 13, 20, 26, 31, 37, 44}
    dots = []
    for i in range(48):
        cx, cy = 248 + (i % 8) * 25, 350 + (i // 8) * 19
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{blue if i in lit else t["line"]}"/>')
    # memory rows: only the name's rows are used.
    rows = {1, 4, 6}
    cells = []
    for r in range(7):
        for c in range(12):
            on = r in rows and c in (3, 4, 5)
            cells.append(f'<rect x="{540 + c * 17}" y="{340 + r * 17}" width="14" height="14" rx="2" '
                         f'fill="#ffffff" opacity="{1 if on else 0.28}"/>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
           'aria-label="The life of the model. Birth: it learns language from scratch on children\'s stories. '
           'Childhood: it learns to use a memory. Life: it reads and learns with no backprop. While it lives, text '
           'goes '
           'to a frozen neural network, a name and a meaning choose a place in the lasting memory, and the memory '
           'nudges the next letter. There is no LLM, stored text or search inside.">',
           f'<rect width="{W}" height="{H}" rx="10" fill="{t["surface"]}"/>',
           text(24, 36, "A neural network with a lasting memory", t["text"], 21, 650),
           text(24, 60, "No LLM inside. No stored text. No search.", t["text2"], 14),
           stage(24, 1, "Birth", ["Starts with random weights.", "Learns language from scratch on",
                                  "100M bytes of children's stories.", "Backprop. 6 hours on a free GPU."]),
           stage(316, 2, "Childhood", ["Language is now frozen.", "Learns how to use a memory,",
                                       "on made-up people and facts.", "Backprop, for the last time."]),
           stage(608, 3, "Life", ["Reads, and learns as it reads.", "No backprop, ever again.",
                                  "Only the memory changes,", "by one simple local rule."], active=True),
           arrow(292, 158, 316, 158), arrow(584, 158, 608, 158),
           text(24, 272, "What happens in life", t["text"], 15, 650),
           panel(24, 176, "1. It reads", ['"Priya lives in', 'a town called', 'Nashik."']),
           panel(224, 240, "2. Neurons fire", ["The network knows language.", "Frozen. It never changes.",
                                               "Reading doesn't touch it."]),
           panel(526, 246, "3. Memory is written", ["The rare name picks the rows.", "The meaning picks the spot.",
                                                    "This is the learning."], fill=blue, ink="#ffffff", ink2="#ffffff"),
           panel(792, 84, "Later", ["Priya ...", "Nashik."]),
           "".join(dots), "".join(cells),
           arrow(200, 430, 224, 430), arrow(464, 430, 526, 430), arrow(772, 430, 792, 430),
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
