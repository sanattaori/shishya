"""Make the README figures (light and dark SVGs) from the benchmark numbers.

    python docs/make_figures.py         # writes docs/img/*.svg

Numbers come from bench/RESULTS.md and train/news_specialist/RESULTS.md; they are written here once so the
figures can be rebuilt without rerunning the benchmarks.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / "img"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {"surface": "#fcfcfb", "card": "#f3f2ef", "line": "#e2e1dc", "text": "#0b0b0b", "text2": "#52514e",
              "muted": "#8a8984", "base": "#a3a29c", "shishya": "#2a78d6", "llm": "#eb6834", "good": "#0ca30c",
              "bad": "#d03b3b", "term": "#16161a", "termtext": "#e8e6e1"},
    "dark": {"surface": "#1a1a19", "card": "#242422", "line": "#383835", "text": "#f0efec", "text2": "#c3c2b7",
             "muted": "#8f8e86", "base": "#6f6e69", "shishya": "#3987e5", "llm": "#d95926", "good": "#0ca30c",
             "bad": "#e66767", "term": "#0f0f12", "termtext": "#e8e6e1"},
}

# (label, value, value text, series)
RULE_BREAKS = [("Small model alone", 38, "38%", "base"), ("Small model + Shishya", 0, "0%", "shishya"),
               ("Big LLM, retry until valid", 0, "0%  (5 s each)", "llm")]
NEWS_F1 = [("Small model (Kev-0.8B)", 0.50, "0.50", "base"), ("FinBERT (2019)", 0.64, "0.64", "base"),
           ("Big LLM (free Nemotron)", 0.78, "0.78", "llm"), ("Shishya news specialist", 0.88, "0.88", "shishya")]
SECONDS = [("Small model + Shishya", 0.20, "0.20 s", "shishya"), ("Big LLM, retry until valid", 5.05, "5.05 s", "llm")]


def text(x, y, s, fill, size=13, weight=400, anchor="start", family=FONT):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family="{family}" xml:space="preserve">{escape(s)}</text>')


def bar(x, y, w, h, fill, r=4):
    """Horizontal bar: square at the baseline, 4px rounded data end."""
    if w <= 0:
        return f'<rect x="{x}" y="{y}" width="2" height="{h}" fill="{fill}"/>'
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} '
            f'Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z" fill="{fill}"/>')


def panel(t, top, title, subtitle, rows, vmax, width=860):
    label_w, left, bar_max, h, gap = 220, 24, 470, 22, 14
    out = [text(left, top, title, t["text"], 15, 600), text(left, top + 19, subtitle, t["text2"], 12)]
    y = top + 36
    x0 = left + label_w
    out.append(f'<line x1="{x0}" y1="{y - 6}" x2="{x0}" y2="{y + len(rows) * (h + gap) - gap + 6}" '
               f'stroke="{t["line"]}" stroke-width="1"/>')
    for label, v, vt, series in rows:
        w = bar_max * v / vmax
        out.append(text(x0 - 12, y + h - 6, label, t["text"], 13, anchor="end"))
        out.append(bar(x0, y, w, h, t[series]))
        out.append(text(x0 + max(w, 2) + 8, y + h - 6, vt, t["text"], 13, 600))
        y += h + gap
    return out, y


def results(theme):
    t = THEMES[theme]
    w = 860
    parts = [f'<rect width="{w}" height="100%" fill="{t["surface"]}"/>']
    # legend (identity is never colour alone: every bar is also labelled)
    lx, ly = 24, 26
    for name, series in (("other small models, no rules", "base"), ("with Shishya", "shishya"),
                         ("big LLM (free)", "llm")):
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="3" fill="{t[series]}"/>')
        parts.append(text(lx + 18, ly, name, t["text2"], 12))
        lx += 26 + 7.2 * len(name)
    y = 62
    for title, sub, rows, vmax in (
            ("Rule breaks", "share of 60 tasks where the answer broke a rule · lower is better", RULE_BREAKS, 40),
            ("Indian news headlines",
             "macro-F1 on the same 300 human-labelled headlines none of them trained on · higher is better",
             NEWS_F1, 1.0),
            ("Time per decision", "median, on an Apple M2 Mac · lower is better", SECONDS, 5.5)):
        p, y = panel(t, y, title, sub, rows, vmax)
        parts += p
        y += 30
    h = y - 8
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="Shishya results: rule breaks 38% to 0%; Indian news macro-F1 0.88 vs 0.78, 0.64 '
            f'(FinBERT) and 0.50; 0.20 s vs 5.05 s per decision">' + "".join(parts) + "</svg>")


def box(t, x, y, w, h, title, lines, accent=None):
    stroke = t[accent] if accent else t["line"]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" stroke="{stroke}" '
           f'stroke-width="{2 if accent else 1}"/>', text(x + 14, y + 24, title, t["text2"], 12, 600)]
    for i, (s, color, weight, family) in enumerate(lines):
        out.append(text(x + 14, y + 50 + i * 22, s, t[color], 14, weight, family=family))
    return out


def arrow(t, x1, x2, y):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" stroke="{t["muted"]}" stroke-width="2"/>'
            f'<path d="M{x2 - 9},{y - 6} L{x2},{y} L{x2 - 9},{y + 6} Z" fill="{t["muted"]}"/>')


def hero(theme):
    t = THEMES[theme]
    w, h = 860, 140
    parts = [f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>']
    y, bh = 14, 112
    boxes = [  # x, width, title, lines, accent
        (12, 180, "AN AGENT WANTS TO RUN", [("rm -rf build/ data/", "text", 600, MONO)], None),
        (222, 170, "SMALL MODEL SAYS", [("run    80%", "bad", 600, MONO), ("ask    15%", "text", 400, MONO),
                                         ("block   5%", "text", 400, MONO)], None),
        (422, 196, "YOUR RULE", [("destructive commands", "text", 400, FONT),
                                 ("always need a person", "text", 400, FONT)], None),
        (648, 200, "SHISHYA ANSWERS", [("ask", "shishya", 700, MONO), ("the best allowed answer", "text2", 400, FONT),
                                       ("+13 ms, 0 rule breaks", "text2", 400, FONT)], "shishya"),
    ]
    for i, (x, bw, title, lines, accent) in enumerate(boxes):
        parts += box(t, x, y, bw, bh, title, lines, accent)
        if i + 1 < len(boxes):
            parts.append(arrow(t, x + bw + 4, boxes[i + 1][0] - 4, y + bh / 2))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="An agent wants to run rm -rf; the small model says run 80%; the rule says destructive '
            f'commands need a person; Shishya answers ask, the best allowed answer">' + "".join(parts) + "</svg>")


TERMINAL = [
    ("$ ", "shishya rules schema.json \"Never buy while the market guard is on.\" --guru claude", "cmd"),
    ("", "  BLOCK  {\"action\": \"add\",  ...}  <- no_add_when_market_guard_on", "bad"),
    ("", "  allow  {\"action\": \"hold\", ...}", "good"),
    ("", "Review these rules, then add them to the schema. Nothing was applied.", "dim"),
    ("", "", "dim"),
    ("$ ", "shishya decide schema.json --input \"ACMEIND: Quarterly profit up 64%\" \\", "cmd"),
    ("", "      --route news_impact=shishya:news-base --backend http://127.0.0.1:8009", "cmd"),
    ("", "status: ok", "out"),
    ("", "answer: {\"news_impact\": \"bullish\", \"action\": \"hold\", \"tell_owner\": true}", "out"),
    ("", "model's top answer: {\"news_impact\": \"bullish\", \"action\": \"add\", ...}", "dim"),
    ("", "broken rule: no_buying_when_market_weak (rejected 1)", "bad"),
    ("", "timings: model_ms 251, search_ms 0", "dim"),
]


def terminal(theme):
    t = THEMES[theme]
    colors = {"cmd": t["termtext"], "out": t["termtext"], "dim": "#9a9890", "bad": "#ff8a7a", "good": "#7ed98b"}
    w, line_h = 860, 21
    h = 46 + line_h * len(TERMINAL) + 14
    parts = [f'<rect width="{w}" height="{h}" rx="10" fill="{t["term"]}"/>']
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        parts.append(f'<circle cx="{20 + i * 18}" cy="18" r="5.5" fill="{c}"/>')
    for i, (prompt, s, kind) in enumerate(TERMINAL):
        y = 52 + i * line_h
        if prompt:
            parts.append(text(18, y, prompt, "#7ab7ff", 13, 600, family=MONO))
        parts.append(text(18 + 16 * bool(prompt), y, s, colors[kind], 13, 400, family=MONO))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Terminal: drafting a rule from English, then a decision that the rule corrected">'
            + "".join(parts) + "</svg>")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"results-{theme}.svg").write_text(results(theme))
        (OUT / f"hero-{theme}.svg").write_text(hero(theme))
        (OUT / f"terminal-{theme}.svg").write_text(terminal(theme))
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
