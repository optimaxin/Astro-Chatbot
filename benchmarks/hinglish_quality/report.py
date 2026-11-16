"""Static HTML dashboard for one or more benchmark runs.

Self-contained (no network): the runs are embedded as JSON and the page
recomputes every number for the selected language filter, so the figures on
screen always match the stored results. With two or more runs the first is
the baseline and the others are compared against it.
"""

from __future__ import annotations

import json
from pathlib import Path

_DIMS = ["grammar", "naturalness", "vocabulary", "meaning", "consistency"]

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hinglish Quality Benchmark</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--good:#1f7a4d;--bad:#b3261e;--accent:#3d5afe;--bar:#c9d3ff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--good:#5cc28d;--bad:#ff8a80;--accent:#8c9eff;--bar:#3a447a}}
:root[data-theme="dark"]{--bg:#151514;--panel:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--good:#5cc28d;--bad:#ff8a80;--accent:#8c9eff;--bar:#3a447a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--muted);font-size:13px}.row{display:flex;flex-wrap:wrap;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;flex:1 1 150px;min-width:140px}
.card .v{font-size:26px;font-weight:650}.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.delta{font-size:13px;font-weight:600}.up{color:var(--good)}.down{color:var(--bad)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.filters button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:6px 14px;cursor:pointer;font:inherit}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.bar{height:8px;background:var(--bar);border-radius:4px}.tablewrap{overflow-x:auto}
.err{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:0 6px;margin:1px 2px;font-size:12px}
.fail{color:var(--bad);font-weight:600}.pass{color:var(--good);font-weight:600}
</style></head><body><main>
<h1>Hinglish Quality Benchmark</h1>
<div class="muted" id="meta"></div>
<div class="filters" id="filters"></div>
<div class="row" id="cards"></div>
<h2>Dimensions (1–5)</h2><div class="tablewrap"><table id="dims"></table></div>
<h2>Most common error categories</h2><div class="tablewrap"><table id="cats"></table></div>
<h2 id="exh">Examples</h2><div class="tablewrap"><table id="rows"></table></div>
<p class="muted">Weights: meaning 0.30, grammar 0.25, vocabulary 0.15, naturalness 0.15, consistency 0.15. A changed or
invented Kundli fact caps a reply at 2.0. Pass = overall ≥ 4.0 and no fact gate. Methodology:
docs/hinglish-quality-benchmark.md.</p>
</main>
<script>
const RUNS = __RUNS__;
const DIMS = __DIMS__;
let lang = "all";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function rowsOf(run){ return run.rows.filter(r => lang === "all" || r.language === lang); }
function stats(rows){
  const n = rows.length || 1, s = {n: rows.length};
  s.overall = rows.reduce((a,r)=>a+r.score.overall,0)/n;
  DIMS.forEach(d => s[d] = rows.reduce((a,r)=>a+r.score.dimensions[d],0)/n);
  s.pass = 100*rows.filter(r=>r.score.passed).length/n;
  s.gate = 100*rows.filter(r=>r.score.gated).length/n;
  s.errors = rows.reduce((a,r)=>a+r.score.errors.length,0);
  s.cats = {}; rows.forEach(r => r.score.errors.forEach(e => s.cats[e.category] = (s.cats[e.category]||0)+1));
  return s;
}
function delta(v, base, fmt, invert){
  if (base === undefined) return "";
  const d = v - base; if (Math.abs(d) < 1e-9) return '<span class="delta">±0</span>';
  const good = invert ? d < 0 : d > 0;
  return `<span class="delta ${good?"up":"down"}">${d>0?"+":""}${fmt(d)}</span>`;
}
function render(){
  const all = RUNS.map(r => stats(rowsOf(r))), base = all[0], last = all[all.length-1], cmp = all.length > 1;
  const f2 = x => x.toFixed(2), f1 = x => x.toFixed(1);
  document.getElementById("meta").innerHTML = RUNS.map(r => `<b>${esc(r.meta.label)}</b> · ${esc(r.meta.mode)} · ${esc(r.meta.created)} · ${esc(r.meta.benchmark_version)} · dataset ${esc(r.meta.dataset_hash)} · judge ${r.meta.judge.used ? esc(r.meta.judge.model) : "off"} · commit ${esc(r.meta.git_commit)}${r.meta.git_dirty?"+dirty":""}`).join("<br>");
  const card = (k, v, d) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>${d||""}</div>`;
  document.getElementById("cards").innerHTML =
    card("Overall quality", f2(last.overall), cmp ? delta(last.overall, base.overall, f2) : "") +
    DIMS.map(d => card(d, f2(last[d]), cmp ? delta(last[d], base[d], f2) : "")).join("") +
    card("Pass rate", f1(last.pass)+"%", cmp ? delta(last.pass, base.pass, f1) : "") +
    card("Errors detected", last.errors, cmp ? delta(last.errors, base.errors, x=>x.toFixed(0), true) : "") +
    card("Fact gate", f1(last.gate)+"%", cmp ? delta(last.gate, base.gate, f1, true) : "");
  document.getElementById("dims").innerHTML = `<tr><th>dimension</th>${RUNS.map(r=>`<th>${esc(r.meta.label)}</th>`).join("")}<th style="width:35%"></th></tr>` +
    ["overall", ...DIMS].map(d => `<tr><td>${d}</td>${all.map(s=>`<td>${f2(s[d])}</td>`).join("")}<td><div class="bar" style="width:${(last[d]/5*100).toFixed(0)}%"></div></td></tr>`).join("");
  const cats = Object.entries(last.cats).sort((a,b)=>b[1]-a[1]).slice(0,12);
  document.getElementById("cats").innerHTML = `<tr><th>category</th>${RUNS.map(r=>`<th>${esc(r.meta.label)}</th>`).join("")}</tr>` +
    (cats.length ? cats.map(([c]) => `<tr><td>${esc(c)}</td>${all.map(s=>`<td>${s.cats[c]||0}</td>`).join("")}</tr>`).join("") : `<tr><td>none</td></tr>`);
  const rows = rowsOf(RUNS[RUNS.length-1]).slice().sort((a,b)=>a.score.overall-b.score.overall);
  document.getElementById("exh").textContent = `Examples — ${RUNS[RUNS.length-1].meta.label} (${rows.length}, lowest first)`;
  document.getElementById("rows").innerHTML = `<tr><th>id</th><th>lang</th><th>score</th><th>reply</th><th>errors</th></tr>` +
    rows.map(r => `<tr><td>${esc(r.id)}<div class="muted">${esc(r.variant)} · ${esc(r.kind)}</div></td><td>${esc(r.language)}</td>
      <td class="${r.score.passed?"pass":"fail"}">${r.score.overall.toFixed(2)}</td>
      <td><div class="muted">${esc(r.question)}</div>${esc(r.text)}</td>
      <td>${r.score.errors.map(e=>`<span class="err" title="${esc(e.message)}">${esc(e.category)}: ${esc(e.span)}</span>`).join("")}</td></tr>`).join("");
}
document.getElementById("filters").innerHTML = ["all","hinglish","hindi","english"].map(l =>
  `<button type="button" data-l="${l}" aria-pressed="${l===lang}">${l === "all" ? "All languages" : l[0].toUpperCase()+l.slice(1)}</button>`).join("");
document.getElementById("filters").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return; lang = b.dataset.l;
  document.querySelectorAll("#filters button").forEach(x => x.setAttribute("aria-pressed", x.dataset.l === lang)); render();
});
render();
</script></body></html>"""


def write_report(paths: list[Path], out: Path) -> Path:
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [{k: r[k] for k in ("id", "variant", "language", "kind", "question", "text", "score")} for r in data["rows"]]
        runs.append({"meta": data["meta"], "rows": rows})
    page = _PAGE.replace("__RUNS__", json.dumps(runs, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__DIMS__", json.dumps(_DIMS))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


__all__ = ["write_report"]
