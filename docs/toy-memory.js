// A tiny version of Shishya's memory, for docs/index.html. It uses the same kind of write rule as brain/v5.py (the delta rule), at a tiny size.
// The real model's reader is replaced by a short word list, and it stores whole words, not letters. It is not the model.
(function (root) {
  const ROWS = 64, ACTIVE = 5, DIM = 32;
  const KINDS = {
    place: ["lives", "live", "town", "city", "home", "village"],
    pet: ["dog", "cat", "pet", "bird", "fish", "rabbit"],
    food: ["food", "eat", "eats", "dish", "meal"],
    work: ["works", "work", "company", "job", "office"],
    friend: ["friend", "friends", "buddy"],
    teacher: ["teacher", "teaches", "school"],
    word: ["word", "secret", "password"],
    other: [],
    change: [],
  };
  const NAMES = Object.keys(KINDS);
  const CHANGE = ["moved", "swapped", "changed", "switched", "got"];
  const COMMON = new Set(("a an and are at best called favourite favorite for has have he her his home in is it its likes "
    + "my named name of on she that the their they to was where who with works lives live town city dog cat pet bird fish "
    + "food company job friend teacher word secret moved swapped changed new now got "
    + "marie john tom anna lily tim sam ben mary jack emma lucy max amy bob").split(" "));

  function hash(s, seed) {
    let h = 2166136261 ^ seed;
    for (const c of s) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function words(text) {
    return text.toLowerCase().replace(/[^a-z' ]/g, " ").split(/\s+/).filter(Boolean).map((w) => w.replace(/'s$|'$/g, ""));
  }
  function unit(v) {
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  }
  function valueCode(word) {
    const v = [];
    for (let i = 0; i < DIM; i++) v.push((hash(word, 1000 + i) / 4294967295) * 2 - 1);
    return unit(v);
  }
  function whoRows(name) {
    const rows = [];
    for (let i = 0; rows.length < ACTIVE; i++) {
      const r = hash(name, i) % ROWS;
      if (!rows.includes(r)) rows.push(r);
    }
    return rows;
  }
  function whatVector(ws) {
    const v = NAMES.map(() => 0);
    let kind = "other";
    for (const n of NAMES) if (KINDS[n].some((k) => ws.includes(k))) { kind = n; break; }
    v[NAMES.indexOf(kind)] = 1;
    const changed = CHANGE.some((c) => ws.includes(c));
    if (changed) {                       // the failure the real model has: a change in new words lands somewhere else
      v[NAMES.indexOf(kind)] = 0.25;
      v[NAMES.indexOf("change")] = 0.97;
    }
    return { w: unit(v), kind: changed ? "change" : kind };
  }

  function createMemory() {
    const M = new Float64Array(ROWS * NAMES.length * DIM);
    const known = new Map();
    const at = (u, i, j) => (u * NAMES.length + i) * DIM + j;
    const k = 1 / Math.sqrt(ACTIVE);

    function read(rows, w) {
      const r = new Array(DIM).fill(0);
      for (const u of rows) for (let i = 0; i < NAMES.length; i++) {
        if (!w[i]) continue;
        for (let j = 0; j < DIM; j++) r[j] += k * w[i] * M[at(u, i, j)];
      }
      return r;
    }
    function teach(sentence) {
      const ws = words(sentence);
      const value = ws[ws.length - 1];
      const name = ws.slice(0, -1).find((x) => !COMMON.has(x) && !NAMES.includes(x));
      if (!name || !value) return { stored: false, value };
      const rows = whoRows(name), { w, kind } = whatVector(ws);
      const target = valueCode(value), r = read(rows, w);
      const err = target.map((t, j) => t - r[j]);
      const strength = Math.min(1, Math.hypot(...err));
      for (const u of rows) for (let i = 0; i < NAMES.length; i++) for (let j = 0; j < DIM; j++) {
        M[at(u, i, j)] += strength * k * w[i] * err[j];       // the delta rule
      }
      known.set(value, target);
      return { stored: true, name, value, rows, kind, strength };
    }
    function ask(question) {
      const ws = words(question);
      const name = ws.find((x) => !COMMON.has(x) && !NAMES.includes(x));
      if (!name) return { found: false, reason: "name" };
      const rows = whoRows(name), { w, kind } = whatVector(ws);
      const r = read(rows, w);
      let best = null, score = 0.5;
      for (const [word, code] of known) {
        const s = code.reduce((a, c, j) => a + c * r[j], 0);
        if (s > score) { best = word; score = s; }
      }
      return best ? { found: true, name, answer: best, rows, kind, score } : { found: false, reason: "empty", name, rows, kind };
    }
    function rowStrength() {
      const out = [];
      for (let u = 0; u < ROWS; u++) {
        let s = 0;
        for (let x = 0; x < NAMES.length * DIM; x++) s += M[at(u, 0, 0) + x] ** 2;
        out.push(Math.sqrt(s));
      }
      return out;
    }
    return { teach, ask, rowStrength };
  }
  root.ToyMemory = { createMemory, ROWS, KINDS: NAMES };
  if (typeof module !== "undefined") module.exports = root.ToyMemory;
})(typeof window !== "undefined" ? window : globalThis);
