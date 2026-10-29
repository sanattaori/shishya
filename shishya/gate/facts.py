"""Facts about a tool call, for the Gate's rules: what it deletes, pushes, reads, sends, and where.

    text, facts = tool_facts("Bash", {"command": "git push --force origin main"}, cwd="/work/app")
    facts["git_force_push"]   # True

Every fact is always present (true/false, or a list), so rules never see a missing fact. Paths are judged
against the project root: the nearest folder above ``cwd`` with a ``.git``, else ``cwd``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .shell import INLINE_FLAGS, INTERPRETERS, SHELLS, Command, Parsed, parse, program

READ_ONLY = {
    "ls", "ll", "la", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep", "rg", "ag", "ack", "find",
    "fd", "tree", "pwd", "echo", "printf", "which", "whereis", "type", "file", "stat", "du", "df", "date", "whoami",
    "id", "uname", "hostname", "ps", "top", "uptime", "diff", "cmp", "sort", "uniq", "cut", "tr", "awk", "jq", "yq",
    "basename", "dirname", "realpath", "readlink", "true", "false", "test", "[", "cd", "sleep", "seq", "nl", "column",
    "md5", "md5sum", "sha1sum", "sha256sum", "shasum", "xxd", "hexdump", "od", "strings", "comm", "tac", "rev",
    "lsof", "free", "vm_stat", "sw_vers", "man", "help", "tldr",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "ls-tree", "describe", "shortlog",
                 "reflog", "grep", "cat-file", "remote", "branch", "tag", "config", "fetch", "stash"}
PROTECTED_BRANCHES = {"main", "master", "production", "prod", "release", "trunk", "live"}
TEMP_PREFIXES = ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/", "/dev/null", "/dev/stdout",
                 "/dev/stderr", "/dev/tty")
SYSTEM_DIRS = {"/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/opt", "/private", "/root", "/sbin", "/srv",
               "/System", "/Library", "/Applications", "/Users", "/usr", "/var", "/Volumes"}
SECRET_NAME = re.compile(
    r"(^|/)(\.env(\.(?!example$|sample$|template$|dist$)[\w.-]+)?|id_(rsa|dsa|ecdsa|ed25519)|[\w.-]+\.(pem|key|p12|pfx)"
    r"|\.netrc|\.pgpass|\.pypirc|\.npmrc|credentials(\.json)?|secrets?\.(ya?ml|json|toml)|\.git-credentials)$")
SECRET_DIRS = ("/.ssh/", "/.aws/", "/.gnupg/", "/.kube/", "/.docker/config.json", "/.config/gcloud/", "/.azure/")
PROD = re.compile(r"(?i)(^|[^a-z0-9])(prod|production|prd|live)([^a-z0-9]|$)")
SQL_DESTRUCTIVE = re.compile(r"(?is)\b(drop\s+(table|database|schema|index|view)|truncate(\s+table)?\s+\w|"
                             r"delete\s+from\s+[\w.\"`]+\s*(;|$)|alter\s+table\s+\S+\s+drop\b|flushall|flushdb|"
                             r"dropdatabase\s*\()")
NETWORK_SEND_FLAGS = {"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "-F", "--form", "-T",
                      "--upload-file", "--post-data", "--post-file", "--json"}
SHELL_CONFIGS = {".bashrc", ".bash_profile", ".zshrc", ".zprofile", ".profile", ".zshenv", ".config/fish/config.fish"}
PUBLISH = {("npm", "publish"), ("yarn", "publish"), ("pnpm", "publish"), ("twine", "upload"), ("uv", "publish"),
           ("cargo", "publish"), ("gem", "push"), ("poetry", "publish"), ("docker", "push"), ("hatch", "publish"),
           ("flit", "publish")}
CLOUD_VERBS = re.compile(r"^(delete|destroy|remove|rm|rb|terminate|purge|drop|uninstall|wipe|reset)([-_].*)?$")
REGENERABLE = {"node_modules", "dist", "build", "out", ".next", ".nuxt", ".turbo", ".cache", ".parcel-cache",
               "coverage", ".nyc_output", "target", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               ".tox", ".venv", "venv", ".gradle", "bin", "obj", "tmp", ".svelte-kit", ".angular", "storybook-static"}
FIND_FILTERS = {"-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-type", "-newer", "-mtime", "-mmin",
                "-size", "-user", "-empty", "-perm", "-wholename"}
FACT_KEYS = (
    "parse_ok", "read_only", "sudo", "deletes", "delete_recursive", "delete_outside_project", "delete_catastrophic",
    "writes_outside_project", "git_force_push", "git_push_protected", "git_delete_remote", "git_discard",
    "db_destructive", "mentions_production", "cloud_destructive", "publishes", "remote_code", "inline_code",
    "runs_script", "reads_secret", "network_send", "secret_exfil", "system_danger", "installs",
    "writes_agent_settings", "writes_shell_config", "writes_git_internals",
)


def project_root(cwd: str) -> Path:
    p = Path(cwd).resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return p


class _Paths:
    def __init__(self, cwd: str, home: str):
        self.cwd = Path(cwd).resolve()
        self.home = Path(home).resolve()
        self.root = project_root(cwd)

    def resolve(self, raw: str) -> Path | None:
        """Where a path argument points; None when it cannot be known (variables, substitutions)."""
        if raw.startswith("~"):
            raw = str(self.home) + raw[1:]
        if "$" in raw or "`" in raw:
            if re.match(r"^\$\{?HOME\}?(/|$)", raw):
                raw = re.sub(r"^\$\{?HOME\}?", str(self.home), raw)
            else:
                return None
        p = Path(raw)
        return Path(os.path.normpath(p if p.is_absolute() else self.cwd / p))

    def inside(self, p: Path) -> bool:
        return p == self.root or self.root in p.parents

    def temp(self, p: Path) -> bool:
        s = str(p)
        return any(s == t.rstrip("/") or s.startswith(t) for t in TEMP_PREFIXES) or s.startswith(
            os.environ.get("TMPDIR", "/nonexistent-tmp").rstrip("/") + "/")

    def outside(self, raw: str) -> bool:
        """True when the path is outside the project (and not a temp dir), or cannot be known."""
        p = self.resolve(raw)
        return p is None or (not self.inside(p) and not self.temp(p))

    def catastrophic(self, raw: str) -> bool:
        """Deleting this would wreck a system, a home folder or the whole project."""
        stripped = raw.rstrip("/") or "/"
        if stripped in ("*", ".*", ".", "..") or re.fullmatch(r"(\.\./)+\.?\.?|\*+", raw):
            return True
        p = self.resolve(raw.rstrip("*").rstrip("/") or "/")
        if p is None:
            return True
        return (str(p) in SYSTEM_DIRS or p == self.home or p == self.root or p in self.root.parents
                or p in self.home.parents or (p.parent == self.home and raw.endswith("*")))


def _flags(words: list[str]) -> set[str]:
    """Short flags split into letters (-rf -> r, f) plus long flags as written."""
    out = set()
    for w in words:
        if w.startswith("--"):
            out.add(w.split("=", 1)[0])
        elif w.startswith("-") and len(w) > 1:
            out.update(f"-{c}" for c in w[1:])
    return out


def _operands(words: list[str]) -> list[str]:
    ops, after_dashdash = [], False
    for w in words[1:]:
        if w == "--" and not after_dashdash:
            after_dashdash = True
        elif after_dashdash or not w.startswith("-") or w == "-":
            ops.append(w)
    return ops


def _secret(raw: str, paths: _Paths) -> bool:
    p = paths.resolve(raw)
    s = str(p) if p is not None else raw
    return bool(SECRET_NAME.search(s)) or any(d in s + ("/" if not s.endswith("/") else "") for d in SECRET_DIRS)


def _git(cmd: Command, f: dict, paths: _Paths) -> bool:
    """Git facts; returns True when the git command only reads."""
    words = cmd.words[1:]
    while words and words[0].startswith("-"):              # git -C dir, -c k=v, --no-pager ...
        words = words[2:] if words[0] in ("-C", "-c", "--git-dir", "--work-tree") else words[1:]
    if not words:
        return True
    sub, args = words[0], words[1:]
    fl = _flags(args)
    if sub == "push":
        refs = [a for a in args if not a.startswith("-")][1:]
        if fl & {"-f", "--force", "--force-with-lease", "--force-if-includes", "--mirror"} or any(
                r.startswith("+") for r in refs):
            f["git_force_push"] = True
        if fl & {"-d", "--delete", "--prune"} or any(r.startswith(":") for r in refs):
            f["git_delete_remote"] = True
        for r in refs:
            dest = r.lstrip("+").split(":")[-1].removeprefix("refs/heads/")
            if dest in PROTECTED_BRANCHES:
                f["git_push_protected"] = True
        return False
    if sub == "reset" and "--hard" in args or sub == "clean" and fl & {"-f", "--force"} \
            or sub == "checkout" and ("--" in args or "." in args or fl & {"-f", "--force"}) \
            or sub == "restore" and not fl & {"--staged", "-S"} or sub in ("filter-branch", "filter-repo") \
            or sub == "branch" and fl & {"-D", "--delete", "-d"} and "-D" in fl | (fl & {"--force", "-f"}) \
            or sub == "stash" and args[:1] in (["drop"], ["clear"]) or sub == "update-ref" and "-d" in args \
            or sub == "reflog" and args[:1] == ["expire"] or sub == "gc" and any(a.startswith("--prune") for a in args):
        f["git_discard"] = True
        return False
    if sub == "branch":
        return not fl & {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy", "-f", "--force"}
    if sub in ("tag", "remote", "config", "stash"):
        return len(args) == 0 or args[0] in ("-l", "--list", "-v", "list", "show", "get", "--get")
    return sub in GIT_READ_ONLY


def _command_facts(cmd: Command, parsed: Parsed, f: dict, paths: _Paths) -> bool:
    """Fill facts from one simple command; returns True when it only reads."""
    prog = program(cmd)
    words = cmd.words
    args = words[1:]
    fl = _flags(args)
    ops = _operands(words)
    lower = " ".join(words).lower()
    reads = prog in READ_ONLY

    for op, target in cmd.redirects:
        if op in (">", ">>", ">|", "<>", "&>"):
            reads = False
            t = target.text
            if t and not t.isdigit() and t != "-":
                if paths.outside(t):
                    f["writes_outside_project"] = True
                _check_written_path(t, f, paths)
        if op in ("<",) and _secret(target.text, paths):
            f["reads_secret"] = True

    if any(_secret(w, paths) for w in ops) and prog not in ("echo", "printf", "git"):
        f["reads_secret"] = True
    if prog in ("env", "printenv", "set", "export") and not ops:
        f["reads_secret"] = True                 # prints the environment, which often holds tokens

    if prog in ("sudo", "doas", "su"):
        f["sudo"] = True
    if prog in ("rm", "rmdir", "unlink", "shred", "srm", "trash", "rimraf", "del", "rd") or (
            prog == "find" and "-delete" in args):
        f["deletes"] = True
        recursive = bool(fl & {"-r", "-R", "--recursive"}) or prog in ("rimraf",) or prog == "find"
        targets = ops if prog != "find" else (ops[:1] or ["."])
        filtered = prog == "find" and any(a in FIND_FILTERS for a in args)
        for t in targets:
            outside = paths.outside(t)
            if outside:
                f["delete_outside_project"] = True
            if recursive and not filtered and not (not outside and _regenerable(t)):
                f["delete_recursive"] = True
            if (recursive or prog == "rm" and "*" in t) and not filtered and paths.catastrophic(t):
                f["delete_catastrophic"] = True
        reads = False
    if prog in ("mv", "cp", "ln", "install", "rsync", "tee", "touch", "mkdir", "chmod", "chown", "truncate", "dd"):
        reads = False
        targets = ops[-1:] if prog in ("mv", "cp", "ln", "install", "rsync") else ops
        if prog == "dd":
            targets = [a[3:] for a in args if a.startswith("of=")]
        for t in targets:
            if ":" in t and prog == "rsync":
                continue
            if paths.outside(t):
                f["writes_outside_project"] = True
            _check_written_path(t, f, paths)
        if prog in ("chmod", "chown") and fl & {"-R", "--recursive"} and any(paths.catastrophic(t) for t in ops[1:]):
            f["system_danger"] = True
        if prog == "mv" and any(paths.catastrophic(t) for t in ops[:-1]):
            f["delete_catastrophic"] = True
        if prog == "dd" and any(t.startswith("/dev/") and not paths.temp(Path(t)) for t in targets):
            f["system_danger"] = True
    if prog in ("shutdown", "reboot", "halt", "poweroff", "init", "launchctl", "systemctl", "diskutil", "fdisk",
                "parted", "wipefs") or prog.startswith("mkfs") or prog in ("kill", "pkill", "killall") and (
            "-1" in args or "1" in ops):
        f["system_danger"] = True
        reads = False

    if prog == "git":
        reads = _git(cmd, f, paths) and reads is not False
        if "push" in args:
            f["network_send"] = True
    elif prog in ("curl", "wget", "http", "https", "xh"):
        if fl & NETWORK_SEND_FLAGS or any(a.startswith(("-d", "--data", "--post")) for a in args) or any(
                a.upper() in ("POST", "PUT", "PATCH", "DELETE") for a in args):
            f["network_send"] = True
            if any(a.startswith("@") and _secret(a[1:], paths) for a in args):
                f["reads_secret"] = True
        if prog == "wget" or fl & {"-o", "-O", "--output"}:
            reads = False
        reads = reads or prog in ("curl", "http", "https", "xh") and not f["network_send"]
        # a download piped into a shell or interpreter runs code from the internet
        for later in parsed.commands:
            if later.pipe_from is not None and parsed.commands[later.pipe_from] is cmd and (
                    program(later) in SHELLS | INTERPRETERS or program(later) in ("sudo", "doas")):
                f["remote_code"] = True
    elif prog in ("scp", "sftp", "ftp", "nc", "ncat", "netcat", "socat", "telnet"):
        f["network_send"] = True
    elif prog == "rsync" and any(":" in o and not o.startswith("/") for o in ops):
        f["network_send"] = True
    elif prog == "ssh" and len(ops) > 1:
        f["network_send"] = True
        reads = False
    elif prog == "gh":
        if args[:2] in (["release", "create"], ["repo", "delete"], ["pr", "merge"], ["gist", "create"]):
            f["publishes" if args[0] != "repo" else "cloud_destructive"] = True
            f["network_send"] = True
        reads = args[:1] in (["status"], ["auth"]) or len(args) > 1 and args[1] in ("view", "list", "status", "diff",
                                                                                   "checks")

    if prog in SHELLS:
        if any("<(" in w and re.search(r"\b(curl|wget)\b", w) for w in args):
            f["remote_code"] = True
        if not any(w == "-c" or re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", w) for w in args) and ops:
            f["runs_script"] = True
        reads = False
    if prog in INTERPRETERS or re.fullmatch(r"python3?(\.\d+)?", prog):
        if fl & INLINE_FLAGS or "-" in ops or cmd.heredoc is not None:
            f["inline_code"] = True
        elif ops:
            f["runs_script"] = True
        reads = False if ops or fl & INLINE_FLAGS else reads
    if prog in ("source", "."):
        f["runs_script"] = True
        reads = False

    if (prog, args[0] if args else "") in PUBLISH:
        f["publishes"] = True
        f["network_send"] = True
    if prog in ("npm", "yarn", "pnpm", "bun", "pip", "pip3", "uv", "brew", "apt", "apt-get", "cargo", "gem", "go",
                "poetry", "conda") and args[:1] and args[0] in ("install", "i", "add", "get"):
        f["installs"] = True
        reads = False

    # databases
    if prog in ("psql", "mysql", "mariadb", "sqlite3", "mongo", "mongosh", "redis-cli", "cockroach", "clickhouse",
                "sqlcmd", "duckdb") and SQL_DESTRUCTIVE.search(" ".join(args) + " " + (cmd.heredoc or "")):
        f["db_destructive"] = True
    if prog in ("dropdb", "dropuser") or (prog == "supabase" and args[:2] == ["db", "reset"]) or (
            prog in ("rails", "rake") and any(a in ("db:drop", "db:reset", "db:schema:load") for a in args)) or (
            prog in ("python", "python3") and "manage.py" in args and "flush" in args) or (
            prog in ("npx", "prisma") and "migrate" in args and "reset" in args) or (
            prog == "redis-cli" and any(a.lower() in ("flushall", "flushdb") for a in args)):
        f["db_destructive"] = True
    if prog in ("psql", "mysql", "mariadb", "mongo", "mongosh", "redis-cli", "sqlcmd") or "db" in args[:2]:
        reads = False

    # cloud and infrastructure
    sub = [a for a in args if not a.startswith("-")]
    if prog in ("terraform", "tofu", "pulumi") and sub[:1] in (["destroy"], ["apply"], ["up"]) and (
            sub[0] == "destroy" or fl & {"-auto-approve", "--auto-approve", "--yes", "-y", "--skip-preview"}):
        f["cloud_destructive"] = True
    if prog in ("kubectl", "helm", "oc") and sub[:1] and sub[0] in ("delete", "uninstall", "drain", "replace", "scale"):
        f["cloud_destructive"] = True
    if prog in ("aws", "gcloud", "az", "doctl", "flyctl", "fly", "heroku", "vercel", "netlify", "railway", "firebase",
                "wrangler") and any(CLOUD_VERBS.match(a) for a in sub[:4]) or (
            prog == "aws" and sub[:2] == ["s3", "rm"]) or (prog == "heroku" and any("destroy" in a for a in sub)):
        f["cloud_destructive"] = True
    if prog in ("docker", "podman") and (sub[:2] in (["system", "prune"], ["volume", "rm"], ["volume", "prune"])
                                         or sub[:1] == ["rm"] and fl & {"-f", "--force", "-v"}):
        f["cloud_destructive"] = True
    if prog in ("terraform", "tofu", "pulumi", "kubectl", "helm", "aws", "gcloud", "az", "docker", "fly", "flyctl",
                "heroku", "vercel", "ansible-playbook") and not reads:
        if PROD.search(lower):
            f["mentions_production"] = True
    if f["db_destructive"] and PROD.search(lower):
        f["mentions_production"] = True
    if f["delete_recursive"] or f["deletes"]:
        reads = False
    return reads


def _regenerable(raw: str) -> bool:
    """A folder tools rebuild (dependencies, build output, caches): deleting it inside the project is routine."""
    name = raw.rstrip("/").rsplit("/", 1)[-1]
    return name in REGENERABLE or name.endswith(".egg-info")


def _check_written_path(raw: str, f: dict, paths: _Paths) -> None:
    p = paths.resolve(raw)
    if p is None:
        return
    s = str(p)
    rel_home = s[len(str(paths.home)) + 1:] if s.startswith(str(paths.home) + "/") else None
    if re.search(r"(^|/)\.claude/(settings(\.local)?\.json|hooks/)|(^|/)\.claude\.json$|/\.shishya/", s):
        f["writes_agent_settings"] = True
    if rel_home in SHELL_CONFIGS:
        f["writes_shell_config"] = True
    if "/.git/" in s + "/" and not s.endswith("/.git/COMMIT_EDITMSG"):
        f["writes_git_internals"] = True
    if _secret(raw, paths):
        f["reads_secret"] = True                      # writing over a secret file also touches it


def shell_facts(command: str, cwd: str, home: str) -> dict:
    paths = _Paths(cwd, home)
    parsed = parse(command)
    f: dict[str, Any] = dict.fromkeys(FACT_KEYS, False)
    f["parse_ok"] = parsed.ok
    all_read = parsed.ok and bool(parsed.commands)
    for cmd in parsed.commands:
        if not _command_facts(cmd, parsed, f, paths):
            all_read = False
    f["read_only"] = all_read
    f["secret_exfil"] = f["reads_secret"] and f["network_send"]
    if re.search(r":\(\)\s*\{.*:\|:", command):          # fork bomb
        f["system_danger"] = True
    f["programs"] = sorted({program(c) for c in parsed.commands if c.argv})
    f["notes"] = parsed.notes
    return f


def _file_facts(tool: str, path: str, cwd: str, home: str) -> dict:
    paths = _Paths(cwd, home)
    f: dict[str, Any] = dict.fromkeys(FACT_KEYS, False)
    f["parse_ok"] = True
    if tool == "Read":
        f["read_only"] = True
        f["reads_secret"] = _secret(path, paths)
    else:
        if paths.outside(path):
            f["writes_outside_project"] = True
        _check_written_path(path, f, paths)
    f["programs"], f["notes"] = [], []
    return f


def _strings(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def tool_facts(tool_name: str, tool_input: dict, cwd: str, home: str | None = None) -> tuple[str, dict]:
    """(the text the model reads, facts) for one tool call."""
    home = home or str(Path.home())
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        return f"Bash: {cmd}", shell_facts(cmd, cwd, home)
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit", "Read"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        root = project_root(cwd)
        try:
            shown = str(Path(path).resolve().relative_to(root))
        except ValueError:
            shown = path
        return f"{tool_name}: {shown}", _file_facts(tool_name, path, cwd, home)
    # MCP and other tools: judge by the tool's name and any SQL or production names in its input
    f: dict[str, Any] = dict.fromkeys(FACT_KEYS, False)
    f["parse_ok"] = True
    verb = tool_name.rsplit("__", 1)[-1].lower()
    blob = " ".join(_strings(tool_input))
    if re.search(r"(^|_)(delete|drop|destroy|remove|purge|truncate|wipe|reset)(_|$)", verb):
        f["cloud_destructive"] = True
    if SQL_DESTRUCTIVE.search(blob):
        f["db_destructive"] = True
    if PROD.search(blob) and (f["db_destructive"] or f["cloud_destructive"] or re.search(
            r"(^|_)(execute|exec|run|apply|deploy|write|update|insert|create)(_|$)", verb)):
        f["mentions_production"] = True
    f["read_only"] = bool(re.search(r"(^|_)(get|list|read|search|query|fetch|describe|view|show)(_|$)", verb)) and \
        not (f["db_destructive"] or f["cloud_destructive"])
    f["programs"], f["notes"] = [], []
    return f"{tool_name}: {blob[:2000]}", f
