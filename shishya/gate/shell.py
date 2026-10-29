"""Split a shell command line into the simple commands it would run, without running anything.

    parse("cd app && rm -rf build/ $(cat list.txt)")
    # -> commands [["cd", "app"], ["rm", "-rf", "build/", <dynamic>], ["cat", "list.txt"]]

Handles quoting, ``;`` ``&&`` ``||`` ``|`` ``&`` and newlines, subshells and groups, ``$(...)``, backticks,
``<(...)``, redirections, heredocs, comments, ``bash -c``/``sh -c``/``eval`` (their strings are parsed too), and
wrappers such as ``sudo``, ``env``, ``nohup``, ``timeout``, ``xargs`` and ``find -exec``.

It is deliberately suspicious: anything it cannot follow (a command name that is a variable or a substitution,
unbalanced quotes, a heredoc fed to an interpreter) sets ``ok = False``, and the Gate then never allows the call on
its own. Words that contain a substitution or a variable are marked dynamic (their value is unknown).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "ash"}
INTERPRETERS = {"python", "python2", "python3", "node", "nodejs", "deno", "bun", "perl", "ruby", "php", "osascript",
                "pwsh", "powershell", "lua", "Rscript"}
INLINE_FLAGS = {"-c", "-e", "-E", "--eval", "-p", "--print", "-r", "-Command", "-command"}
MAX_DEPTH = 8

_OPS = ("&&", "||", "|&", ";;", ";", "|", "&", "\n")
_REDIR = re.compile(r"(\d*|&)(>>|>\||>&|>|<<<|<<-|<<|<>|<&|<)")


@dataclass
class Word:
    text: str                 # the word with quotes removed (substitutions kept as written)
    dynamic: bool = False     # contains a variable or a command substitution: the real value is unknown
    quoted: bool = False      # was (partly) quoted, so globs and ~ in it are literal


@dataclass
class Command:
    argv: list[Word]
    redirects: list[tuple[str, Word]] = field(default_factory=list)   # (operator, target)
    pipe_from: int | None = None          # index (in Parsed.commands) of the command piping into this one
    heredoc: str | None = None            # body of a heredoc fed to this command
    inner: bool = False                   # came from inside $(...), backticks, bash -c, eval, find -exec, ...

    @property
    def words(self) -> list[str]:
        return [w.text for w in self.argv]


@dataclass
class Parsed:
    commands: list[Command] = field(default_factory=list)
    ok: bool = True
    notes: list[str] = field(default_factory=list)

    def fail(self, why: str) -> None:
        self.ok = False
        if why not in self.notes:
            self.notes.append(why)


def _matching(s: str, i: int, open_: str, close: str) -> int:
    """Index just after the ``close`` matching the ``open_`` that ends at ``i`` (quotes respected), or -1."""
    depth, j, n = 1, i, len(s)
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "'":
            k = s.find("'", j + 1)
            if k < 0:
                return -1
            j = k + 1
            continue
        if c == '"':
            j = _end_double(s, j + 1)
            if j < 0:
                return -1
            continue
        if s.startswith(open_, j):
            depth += 1
            j += len(open_)
            continue
        if c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _end_double(s: str, j: int) -> int:
    """Index just after the closing double quote of a string starting at ``j`` (inside the quotes), or -1."""
    n = len(s)
    while j < n:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return j + 1
        if s.startswith("$(", j):
            k = _matching(s, j + 2, "(", ")")
            if k < 0:
                return -1
            j = k
            continue
        if c == "`":
            k = s.find("`", j + 1)
            if k < 0:
                return -1
            j = k + 1
            continue
        j += 1
    return -1


class _Lexer:
    """Turns a command line into tokens: ("word", Word) or ("op", str) or ("redir", str)."""

    def __init__(self, s: str, out: Parsed, depth: int):
        self.s, self.out, self.depth = s, out, depth
        self.subs: list[str] = []        # command strings found inside $(...), backticks and <(...)

    def tokens(self) -> list[tuple[str, object]]:
        s, n, i = self.s, len(self.s), 0
        toks: list[tuple[str, object]] = []
        while i < n:
            c = s[i]
            if c in " \t\r":
                i += 1
                continue
            if c == "\\" and i + 1 < n and s[i + 1] == "\n":       # line continuation
                i += 2
                continue
            if c == "#":                                              # comment to end of line
                k = s.find("\n", i)
                i = n if k < 0 else k
                continue
            op = next((o for o in _OPS if s.startswith(o, i)), None)
            if op:
                toks.append(("op", op))
                i += len(op)
                continue
            if c in "(){}" and (c in "()" or self._standalone(i)):
                if c == "(" and s.startswith("((", i):              # arithmetic: no commands inside
                    k = s.find("))", i)
                    i = n if k < 0 else k + 2
                    continue
                toks.append(("op", c))
                i += 1
                continue
            m = _REDIR.match(s, i)
            if m and not (c.isdigit() and m.group(1) == "" and m.group(2) == ""):
                toks.append(("redir", m.group(2)))
                i = m.end()
                continue
            i = self._word(i, toks)
        return toks

    def _standalone(self, i: int) -> bool:
        s = self.s
        before = s[i - 1] if i > 0 else " "
        after = s[i + 1] if i + 1 < len(s) else " "
        return before in " \t\n;&|(" and after in " \t\n;&|)"

    def _word(self, i: int, toks: list) -> int:
        s, n = self.s, len(self.s)
        text, dynamic, quoted = [], False, False
        while i < n:
            c = s[i]
            if c in " \t\r\n;&|" or (c in "()" and not text) or c == ")":
                break
            if c == "<" and s.startswith("<(", i) or c == ">" and s.startswith(">(", i):   # process substitution
                k = _matching(s, i + 2, "(", ")")
                if k < 0:
                    self.out.fail("unbalanced process substitution")
                    return n
                self.subs.append(s[i + 2:k - 1])
                text.append(s[i:k])
                dynamic, i = True, k
                continue
            if c in "<>" and text:
                break
            if c == "\\":
                text.append(s[i + 1:i + 2])
                i += 2
                continue
            if c == "'":
                k = s.find("'", i + 1)
                if k < 0:
                    self.out.fail("unbalanced single quote")
                    return n
                text.append(s[i + 1:k])
                quoted, i = True, k + 1
                continue
            if c == '"':
                k = _end_double(s, i + 1)
                if k < 0:
                    self.out.fail("unbalanced double quote")
                    return n
                inner = s[i + 1:k - 1]
                d = self._scan_dollars(inner)
                text.append(re.sub(r'\\([\\"$`])', r"\1", inner))
                dynamic, quoted, i = dynamic or d, True, k
                continue
            if c == "`":
                k = s.find("`", i + 1)
                if k < 0:
                    self.out.fail("unbalanced backtick")
                    return n
                self.subs.append(s[i + 1:k])
                text.append(s[i:k + 1])
                dynamic, i = True, k + 1
                continue
            if c == "$":
                k = self._dollar(i)
                if k < 0:
                    return n
                text.append(s[i:k])
                dynamic, i = True, k
                continue
            text.append(c)
            i += 1
        toks.append(("word", Word("".join(text), dynamic, quoted)))
        return i

    def _dollar(self, i: int) -> int:
        """Skip a $-expansion starting at ``i``; command substitutions are collected. Returns the end, or -1."""
        s = self.s
        if s.startswith("$((", i):
            k = s.find("))", i)
            return len(s) if k < 0 else k + 2
        if s.startswith("$(", i):
            k = _matching(s, i + 2, "(", ")")
            if k < 0:
                self.out.fail("unbalanced $(")
                return -1
            self.subs.append(s[i + 2:k - 1])
            return k
        if s.startswith("${", i):
            k = s.find("}", i)
            if k < 0:
                self.out.fail("unbalanced ${")
                return -1
            return k + 1
        m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])", s[i:])
        return i + (m.end() if m else 1)

    def _scan_dollars(self, inner: str) -> bool:
        """Collect substitutions inside a double-quoted string; True when it expands anything."""
        dyn, j = False, 0
        while j < len(inner):
            if inner[j] == "\\":
                j += 2
                continue
            if inner.startswith("$(", j) and not inner.startswith("$((", j):
                k = _matching(inner, j + 2, "(", ")")
                if k < 0:
                    self.out.fail("unbalanced $( in quotes")
                    return True
                self.subs.append(inner[j + 2:k - 1])
                dyn, j = True, k
                continue
            if inner[j] == "`":
                k = inner.find("`", j + 1)
                if k < 0:
                    self.out.fail("unbalanced backtick in quotes")
                    return True
                self.subs.append(inner[j + 1:k])
                dyn, j = True, k + 1
                continue
            if inner[j] == "$" and j + 1 < len(inner) and (inner[j + 1].isalnum() or inner[j + 1] in "_{@*#?$!-"):
                dyn = True
            j += 1
        return dyn


_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_.-]*)\2")


def _split_heredocs(command: str) -> tuple[str, list[tuple[str, str]]]:
    """Cut heredoc bodies out of the text before lexing: (text with bodies blanked, [(delimiter, body), ...])."""
    lines = command.split("\n")
    bodies: list[tuple[str, str]] = []
    out_lines: list[str] = []
    n = 0
    while n < len(lines):
        line = lines[n]
        out_lines.append(line)
        n += 1
        for m in _HEREDOC.finditer(line):
            if line[m.start():].startswith("<<<") or (m.start() and line[m.start() - 1] == "<"):
                continue                        # a here-string, not a heredoc
            strip, delim = m.group(1) == "-", m.group(3)
            body = []
            while n < len(lines):
                text = lines[n].lstrip("\t") if strip else lines[n]
                out_lines.append("")            # keep line numbers; the body is not commands
                n += 1
                if text == delim:
                    break
                body.append(text)
            else:
                bodies.append((delim, None))    # no end marker
                continue
            bodies.append((delim, "\n".join(body)))
    return "\n".join(out_lines), bodies


def parse(command: str, *, _out: Parsed | None = None, _depth: int = 0, _inner: bool = False) -> Parsed:
    """The simple commands in ``command`` (including those inside substitutions and ``bash -c`` strings)."""
    out = _out if _out is not None else Parsed()
    if _depth > MAX_DEPTH:
        out.fail("nested too deeply")
        return out
    first = len(out.commands)
    text, bodies = _split_heredocs(command)
    lex = _Lexer(text, out, _depth)
    toks = lex.tokens()
    heredoc_cmds: list[tuple[Command, str]] = []
    cur: Command | None = None
    k = 0
    while k < len(toks):
        kind, val = toks[k]
        if kind == "op":
            if cur is not None and cur.argv:
                out.commands.append(cur)
            cur = None
            if val in ("|", "|&"):
                cur = Command([], pipe_from=len(out.commands) - 1, inner=_inner)
            k += 1
            continue
        if cur is None:
            cur = Command([], inner=_inner)
        if kind == "redir":
            target = toks[k + 1][1] if k + 1 < len(toks) and toks[k + 1][0] == "word" else Word("", True)
            if val in ("<<", "<<-"):
                heredoc_cmds.append((cur, target.text))
            else:
                cur.redirects.append((val, target))
            k += 2
            continue
        cur.argv.append(val)  # type: ignore[arg-type]
        k += 1
    if cur is not None and cur.argv:
        out.commands.append(cur)
    # every heredoc operator the lexer saw must match a body cut out above, in order; otherwise we cannot tell
    # commands from data, so we fail closed
    if len(heredoc_cmds) != len(bodies):
        out.fail("heredoc that could not be matched to its body")
    for (cmd, delim), (want, body) in zip(heredoc_cmds, bodies):
        if body is None or delim.strip("'\"") != want:
            out.fail("heredoc without its end marker")
        else:
            cmd.heredoc = body
    for sub in lex.subs:
        parse(sub, _out=out, _depth=_depth + 1, _inner=True)
    # expand what wrappers run: bash -c, eval, sudo, env, xargs, find -exec, ... (new commands are appended)
    i = first
    while i < len(out.commands):
        cmd = out.commands[i]
        if not getattr(cmd, "_expanded", False):
            cmd._expanded = True  # type: ignore[attr-defined]
            _expand(cmd, out, _depth)
        i += 1
    return out


def program(cmd: Command) -> str:
    """The command name without its directory (``/usr/bin/rm`` -> ``rm``)."""
    return cmd.argv[0].text.rsplit("/", 1)[-1] if cmd.argv else ""


_WRAPPERS_WITH_ARG = {"timeout": 1, "nice": 0, "ionice": 0, "stdbuf": 0, "chroot": 1, "watch": 0, "taskset": 1}
_PLAIN_WRAPPERS = {"sudo", "doas", "nohup", "time", "command", "builtin", "exec", "env", "caffeinate", "unbuffer",
                   "npx", "bunx", "pnpx", "uvx", "pipx", "xargs", "parallel", "su"}


def _expand(cmd: Command, out: Parsed, depth: int) -> None:
    """Add the commands a wrapper runs. Also checks that the command name itself is known."""
    argv = cmd.argv
    j = 0                               # leading VAR=value assignments: the command is what follows
    while j < len(argv) and not argv[j].quoted and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[j].text):
        j += 1
    if j == len(argv):
        return
    if j:
        cmd.argv = argv = argv[j:]
    if argv[0].dynamic:
        out.fail(f"the command name {argv[0].text!r} is only known when it runs")
        return
    prog = program(cmd)
    words = cmd.words
    if prog in SHELLS:
        k = next((i for i, w in enumerate(words[1:], 1)
                  if w == "-c" or (re.fullmatch(r"-[a-zA-Z]+", w) and "c" in w)), None)
        if k is not None:
            if k + 1 >= len(words):
                out.fail(f"{prog} -c without a command string")
            else:
                if argv[k + 1].dynamic:
                    out.fail(f"{prog} -c runs a string that is only known when it runs")
                parse(words[k + 1], _out=out, _depth=depth + 1, _inner=True)
        elif cmd.heredoc is not None:
            parse(cmd.heredoc, _out=out, _depth=depth + 1, _inner=True)
        elif cmd.pipe_from is not None or len(words) == 1 or any(r[0] in ("<", "<<<") for r in cmd.redirects):
            out.fail(f"{prog} runs commands that are piped or redirected into it")
        return
    if prog == "eval":
        joined = " ".join(words[1:])
        if any(w.dynamic for w in argv[1:]):
            out.fail("eval runs a string that is only known when it runs")
        parse(joined, _out=out, _depth=depth + 1, _inner=True)
        return
    if prog in INTERPRETERS and cmd.heredoc is not None:
        out.fail(f"{prog} runs code from a heredoc")
        return
    if prog in _PLAIN_WRAPPERS or prog in _WRAPPERS_WITH_ARG:
        rest = _after_wrapper(prog, argv)
        if rest:
            if prog in ("xargs", "parallel"):
                out.fail(f"{prog} adds arguments that are only known when it runs")
            _add(out, Command(rest, cmd.redirects, cmd.pipe_from, cmd.heredoc, True), depth)
        return
    if prog == "find":
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            for k, w in enumerate(words):
                if w == flag:
                    end = next((m for m in range(k + 1, len(words)) if words[m] in (";", "+", "\\;")), len(words))
                    if end > k + 1:
                        _add(out, Command(argv[k + 1:end], [], None, None, True), depth)
        return
    if prog == "git":
        # git -c alias / git config could run anything, but we do not try: the git facts look at subcommands
        return


def _add(out: Parsed, cmd: Command, depth: int) -> None:
    if depth > MAX_DEPTH:
        out.fail("nested too deeply")
        return
    out.commands.append(cmd)


def _flag_value(words: list[str], flags: tuple[str, ...]) -> str | None:
    for k, w in enumerate(words):
        if w in flags and k + 1 < len(words):
            return words[k + 1]
    return None


def _after_wrapper(prog: str, argv: list[Word]) -> list[Word]:
    """The command a wrapper runs (its own options and arguments skipped)."""
    k = 1
    n = len(argv)
    if prog == "env":
        while k < n and (argv[k].text.startswith("-") or "=" in argv[k].text):
            if argv[k].text in ("-u", "-C", "-S") and k + 1 < n:
                k += 1
            k += 1
        return argv[k:]
    if prog in ("sudo", "doas", "su"):
        takes_arg = {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "--user", "--group"}
        while k < n and argv[k].text.startswith("-"):
            if argv[k].text in takes_arg and k + 1 < n:
                k += 1
            k += 1
        if prog == "su":
            c = next((i for i in range(k, n) if argv[i].text == "-c"), None)
            return [Word("sh"), Word("-c"), argv[c + 1]] if c is not None and c + 1 < n else []
        return argv[k:]
    if prog in ("xargs", "parallel"):
        takes_arg = {"-n", "-P", "-L", "-I", "-s", "-d", "-E", "-a", "--max-args", "--max-procs", "--delimiter",
                     "-j", "--jobs"}
        while k < n and argv[k].text.startswith("-"):
            if argv[k].text in takes_arg and k + 1 < n:
                k += 1
            k += 1
        return argv[k:]
    if prog in ("npx", "bunx", "pnpx", "uvx", "pipx"):
        if prog == "pipx" and k < n and argv[k].text == "run":
            k += 1
        while k < n and argv[k].text.startswith("-"):
            if argv[k].text in ("--from", "-p", "--package", "--with", "--python") and k + 1 < n:
                k += 1
            k += 1
        return argv[k:]
    if prog in _WRAPPERS_WITH_ARG:
        while k < n and argv[k].text.startswith("-"):
            if argv[k].text in ("-n", "-c", "-o", "-e", "-i", "-s", "-k") and k + 1 < n:
                k += 1
            k += 1
        k += _WRAPPERS_WITH_ARG[prog]
        return argv[k:]
    while k < n and argv[k].text.startswith("-"):
        k += 1
    return argv[k:]
