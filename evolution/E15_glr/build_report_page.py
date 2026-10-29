"""build_report_page.py OUT_HTML --sweep sweep_runs.csv --manifest E15_ood_manifest.json [--funnel funnel.csv] [--parallax parallax_index.jsonl]
        [--figs DIR] [--frames-root RUN/bench/frames] [--demo demo_summary.json] [--stats E15_stats.json] [--fragment OUT_FRAGMENT.html] [--md OUT.md]

Builds the self-contained summary page of the E15 sandbox test (Chinese): tables and figures come from the CSV / json outputs of the other scripts in this
directory, the numbers in the prose are computed from them, so re-running after new results refreshes the whole page. Everything is inlined (CSS, PNG
figures and thumbnails as data: URIs): the file opens offline. Thumbnails are only made for clips whose licence allows showing a frame (not CC BY / CC BY-SA /
restricted), the rest get a text tile.
OUT_HTML is a complete document (works from the repo); --fragment writes the same page without the <html>/<head> wrapper (what the Artifact tool wants).
"""
import argparse
import base64
import csv
import html
import io
import json
import statistics
from html.parser import HTMLParser
from pathlib import Path

REPO_URL = "https://github.com/Xiaoxuxu994/3ddata_anno/blob/e15-glr-track"
NO_THUMB = ("CC BY", "CC BY-SA", "Restricted")


def esc(x):
    return html.escape("" if x is None else str(x), quote=True)


def num(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if v != v else f"{v:.{nd}f}"


def read_csv(p):
    if not p or not Path(p).exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def thumb(frames_root, clip, width=288):
    p = Path(frames_root) / clip / "000090.jpg"
    if not p.exists():
        return None
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=68, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def badge(kind, text):
    icon = {"good": "✓", "warn": "!", "crit": "✕", "info": "·"}[kind]
    return f'<span class="b b-{kind}"><i aria-hidden="true">{icon}</i>{esc(text)}</span>'


def table(head, rows, cls="", note=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw {cls}"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>' + (f'<p class="note">{note}</p>' if note else "")


def figure(figs, name, cap, alt):
    p = Path(figs) / name if figs else None
    if not p or not p.exists():
        return f'<p class="pending">图 {esc(name)} 还没有生成（数据未齐）。</p>'
    return f'<figure><img src="{b64(p, "image/png")}" alt="{esc(alt)}" data-file="{esc(name)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


CSS = """
:root{--bg:#f4f6f8;--surface:#fff;--ink:#14202b;--ink2:#46576a;--muted:#6f7f8f;--line:#dbe2e9;--accent:#175c9c;--accent-soft:#e3eefa;
--good:#0b7a0b;--good-bg:#e4f4e4;--warn:#8a5a00;--warn-bg:#fcf0d2;--crit:#b0282a;--crit-bg:#fbe5e5;--info-bg:#e9eef3;--code:#eef2f6;--fig:#fcfcfb;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--shadow:0 1px 2px rgba(20,32,43,.06)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1319;--surface:#161d26;--ink:#e6edf3;--ink2:#aebccb;--muted:#8393a3;--line:#27323f;--accent:#6db3ef;--accent-soft:#16283a;
--good:#5fd35f;--good-bg:#12301a;--warn:#f2c14e;--warn-bg:#3a2d0c;--crit:#ff8080;--crit-bg:#3d1618;--info-bg:#1d2733;--code:#1b2530;--shadow:none;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1319;--surface:#161d26;--ink:#e6edf3;--ink2:#aebccb;--muted:#8393a3;--line:#27323f;--accent:#6db3ef;--accent-soft:#16283a;
--good:#5fd35f;--good-bg:#12301a;--warn:#f2c14e;--warn-bg:#3a2d0c;--crit:#ff8080;--crit-bg:#3d1618;--info-bg:#1d2733;--code:#1b2530;--shadow:none;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
padding-block:0 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:960px;margin:0 auto}
a{color:var(--accent);text-underline-offset:2px}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.88em}
code{background:var(--code);padding:1px 5px;border-radius:4px}
.nav{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin:0 calc(-1*max(16px,env(safe-area-inset-left,0px)));
padding:0 max(16px,env(safe-area-inset-left,0px));overflow-x:auto;white-space:nowrap}
.nav a{display:inline-block;padding:11px 12px 9px;color:var(--ink2);text-decoration:none;font-size:13.5px;border-bottom:2px solid transparent}
.nav a:hover,.nav a:focus-visible{color:var(--accent);border-bottom-color:var(--accent);outline:none}
header.mast{padding:40px 0 22px}
.eyebrow{font:600 12px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font-size:clamp(28px,5vw,42px);line-height:1.2;margin:12px 0 10px;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--ink2);font-size:16.5px;max-width:46em;margin:0 0 16px}
.facts{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px;margin:0 0 16px;padding:0;list-style:none}
.caution{background:var(--warn-bg);color:var(--ink);border-left:3px solid var(--warn);padding:10px 14px;border-radius:0 6px 6px 0;font-size:14px;margin:0}
section{padding-top:36px}
h2{font-size:22px;line-height:1.3;margin:0 0 6px;text-wrap:balance}
h2 small{display:block;font:500 13px/1.5 ui-monospace,Menlo,Consolas,monospace;color:var(--muted);letter-spacing:.02em;margin-top:2px}
h3{font-size:16px;margin:22px 0 6px}
p{margin:8px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0 6px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow);min-width:0}
.kpi b{display:block;font-size:30px;line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.kpi span{display:block;color:var(--ink2);font-size:13.5px;margin-top:6px}
.kpi.crit b{color:var(--crit)}.kpi.good b{color:var(--good)}.kpi.warn b{color:var(--warn)}
.find{display:grid;gap:10px;margin:12px 0}
.find>div{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:0}
.find strong{display:block;margin-bottom:2px}
.tw{overflow-x:auto;margin:10px 0;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--ink2);background:var(--info-bg);white-space:nowrap;position:sticky;top:0}
tr:last-child td{border-bottom:none}
td{font-variant-numeric:tabular-nums}
.tw.tall{max-height:560px;overflow:auto}
.note,.pending{color:var(--muted);font-size:13px;margin:4px 0 0}
.pending{background:var(--info-bg);padding:8px 12px;border-radius:6px}
.b{display:inline-flex;align-items:center;gap:4px;padding:1px 8px 1px 6px;border-radius:999px;font-size:12px;line-height:1.6;white-space:nowrap}
.b i{font-style:normal;font-weight:700}
.b-good{background:var(--good-bg);color:var(--good)}.b-warn{background:var(--warn-bg);color:var(--warn)}.b-crit{background:var(--crit-bg);color:var(--crit)}.b-info{background:var(--info-bg);color:var(--ink2)}
figure{margin:14px 0;background:var(--fig);border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{display:block;width:100%;height:auto}
figcaption{color:#46576a;font-size:13px;padding:6px 6px 2px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;counter-reset:s;margin:14px 0;padding:0;list-style:none}
.steps li{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:13.5px;min-width:0;position:relative}
.steps li b{display:block;font-size:14.5px;margin-bottom:2px}
.steps li.new{border-color:var(--accent);background:var(--accent-soft)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-width:0;font-size:12.5px}
.card .ph{aspect-ratio:16/9;background:var(--info-bg);display:flex;align-items:center;justify-content:center;color:var(--muted);text-align:center;padding:6px;font-size:11.5px}
.card img{display:block;width:100%;height:100%;object-fit:cover}
.card .tx{padding:6px 8px 8px;display:flex;flex-direction:column;gap:4px}
.card .id{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;overflow-wrap:anywhere}
.card .bs{display:flex;flex-wrap:wrap;gap:3px}
details{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 14px;margin:12px 0}
summary{cursor:pointer;font-weight:600}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
ul.tight{margin:6px 0;padding-left:1.3em}ul.tight li{margin:3px 0}
footer{margin-top:48px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:16px}
@media (max-width:560px){body{font-size:14.5px}.kpi b{font-size:26px}}
"""


class _MD(HTMLParser):
    """The subset of HTML this page uses -> GitHub Markdown (pipe tables, figures linked from fig_dir, badges as plain text, no thumbnails)."""

    def __init__(self, fig_dir):
        super().__init__(convert_charrefs=True)
        self.out, self.fig_dir = [], fig_dir
        self.stack, self.skip, self.row, self.cell, self.rows, self.in_table = [], 0, None, None, [], False
        self.list_stack = []

    def w(self, t):
        if self.cell is not None:
            self.cell.append(t)
        else:
            self.out.append(t)

    def handle_starttag(self, tag, at):
        at = dict(at)
        cls = at.get("class", "") or ""
        if tag in ("style", "title", "script", "nav") or "gal" in cls.split() or "ph" in cls.split():
            self.skip += 1
            self.stack.append((tag, True))
            return
        self.stack.append((tag, False))
        if self.skip:
            return
        if tag == "h1":
            self.w("\n# ")
        elif tag == "h2":
            self.w("\n\n## ")
        elif tag == "h3":
            self.w("\n\n### ")
        elif tag == "small":
            self.w(" _(")
        elif tag in ("p", "figure") or (tag == "div" and "kpis" in cls) or (tag == "div" and "find" in cls):
            self.w("\n\n")
        elif tag == "div" and ("kpi" in cls.split() or tag == "div" and self.stack and False):
            self.w("\n- ")
        elif tag in ("ul", "ol"):
            self.list_stack.append([tag, 0])
            self.w("\n")
        elif tag == "li":
            kind = self.list_stack[-1] if self.list_stack else ["ul", 0]
            kind[1] += 1
            self.w("\n" + (f"{kind[1]}. " if kind[0] == "ol" else "- "))
        elif tag in ("code",):
            self.w("`")
        elif tag in ("b", "strong"):
            self.w("**")
        elif tag == "a":
            self.href = at.get("href", "")
            self.w("[")
        elif tag == "img":
            f = at.get("data-file")
            if f:
                self.w(f"![{at.get('alt', '')}]({self.fig_dir}/{f})")
        elif tag == "table":
            self.rows, self.in_table = [], True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "summary":
            self.w("\n\n**")
        elif tag == "figcaption":
            self.w("\n\n*")
        elif tag == "br":
            self.w(" ")

    def handle_endtag(self, tag):
        top = self.stack.pop() if self.stack else (tag, False)
        if top[1]:
            self.skip -= 1
            return
        if self.skip:
            return
        if tag == "small":
            self.w(")_")
        elif tag in ("code",):
            self.w("`")
        elif tag in ("b", "strong"):
            self.w("**")
        elif tag == "summary":
            self.w("**\n")
        elif tag == "a":
            self.w(f"]({getattr(self, 'href', '')})")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.w("\n")
        elif tag in ("td", "th"):
            txt = "".join(self.cell).replace("|", "\\|").replace("\n", " ").strip()
            self.cell = None
            self.row.append(txt)
        elif tag == "tr":
            self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            if self.rows:
                head, body = self.rows[0], self.rows[1:]
                self.out.append("\n\n| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n")
                self.out.extend("| " + " | ".join(r) + " |\n" for r in body)
            self.in_table = False
        elif tag == "figcaption":
            self.w("*\n")
        elif tag == "p":
            self.w("\n")

    def handle_data(self, d):
        if self.skip:
            return
        if self.cell is None and not d.strip():
            return
        import re
        self.w(re.sub(r"\s+", " ", d))


def html_to_md(fragment, fig_dir="figures"):
    c = _MD(fig_dir)
    c.feed(fragment)
    import re
    md = "".join(c.out)
    md = re.sub(r"[ \t]*\n[ \t]*", "\n", md)          # whitespace around line breaks comes from the HTML source, not from the page
    md = re.sub(r" {2,}", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    # CommonMark does not close "**...**" before a CJK character when the text inside ends in punctuation: a space after the closing mark fixes that
    md = re.sub(r"(\*\*[^*\n]+?\*\*)(?=[^\s.,;:)\]\-*，。；：、）])", r"\1 ", md)
    return md


def build(a):
    sweep = read_csv(a.sweep)
    man = json.loads(Path(a.manifest).read_text())["clips"]
    fun = {r["clip"]: r for r in read_csv(a.funnel)}
    par = {}
    if a.parallax and Path(a.parallax).exists():
        for ln in Path(a.parallax).read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                par[d["clip"].replace(".mp4", "")] = d
    stats = json.loads(Path(a.stats).read_text()) if a.stats and Path(a.stats).exists() else {}
    demo = json.loads(Path(a.demo).read_text()) if a.demo and Path(a.demo).exists() else []
    by = {}
    for r in sweep:
        by.setdefault(r["clip"], {})[r["run"]] = r
    easy = [c for c in man if c["role"] == "easy"]
    ctl = [c for c in man if c["role"] == "control_negative"]
    ids_e, ids_c = [c["id"] for c in easy], [c["id"] for c in ctl]
    n_local = sum(1 for c in ctl if c.get("local_only"))
    v0 = lambda c: by.get(c, {}).get("v0a")  # noqa: E731

    # ---- numbers
    e_rows = [v0(c) for c in ids_e if v0(c)]
    c_rows = [v0(c) for c in ids_c if v0(c)]
    e_model = [r for r in e_rows if r.get("n_points")]
    e_pass = [r for r in e_rows if r["verdict"] in ("ok", "suspect")]
    c_stop = [r for r in c_rows if r["verdict"] == "stop"]
    scored = [r for c in ids_e for r in [by.get(c, {}).get("v0a")] if r and r.get("outcome")]
    n_silver = sum(r["silver_accurate"] == "True" for r in scored)
    n_silent = sum(r["outcome"] == "silent_error" for r in scored)
    n_true_pass = sum(r["outcome"] == "true_pass" for r in scored)
    n_false_stop = sum(r["outcome"] == "false_stop" for r in scored)
    fe = [abs(float(r["focal_err_pct"])) for r in scored if r.get("focal_err_pct")]
    n_f10 = sum(x > 10 for x in fe)
    ate = [float(r["ate_vs_ref_pct"]) for r in scored if r.get("ate_vs_ref_pct")]
    n_ate_ok = sum(x <= 3.27 for x in ate)
    ref_failed = [r["clip"] for c in ids_e for r in [by.get(c, {}).get("v0a")] if r and r.get("ref_failed") == "True"]
    # filter layers
    f_easy = [fun[c] for c in ids_e if c in fun]
    f_ctl = [fun[c] for c in ids_c if c in fun]
    fe_pass = sum(1 for r in f_easy if r["first_stop"] in ("gate", "passed_all", "incomplete") and r["rules"] != "stop" and r["vlm"] != "stop")
    fc_caught = sum(1 for r in f_ctl if r["first_stop"] in ("rules", "vlm", "route", "gate"))
    single = [r["clip"] for r in f_ctl if r.get("n_layers_stopping") == "1"]
    t_easy_par = [par[c]["rot_p90_over_flow_med"] for c in ids_e if c in par and par[c].get("rot_p90_over_flow_med") is not None]
    pans = [par[c]["rot_p90_over_flow_med"] for c in ("ctl_pan_4448", "mk_3110") if c in par and par[c].get("rot_p90_over_flow_med") is not None]

    def pending(txt="待补"):
        return f'<span class="note">{txt}</span>'

    # ---- gallery
    def card(c):
        r = v0(c["id"]) or {}
        lic = c.get("license") or ""
        t = None if any(k in lic for k in NO_THUMB) or not a.frames_root else thumb(a.frames_root, c["id"])
        lic_txt = "仅限个人使用，只在本地试过" if lic.startswith("Restricted") else lic
        ph = f'<img src="{t}" alt="{esc(c["id"])} 的第 90 帧" loading="lazy">' if t else f'<span>{esc(lic_txt)}<br>（不展示画面）</span>'
        bs = []
        if r:
            v = r["verdict"]
            bs.append(badge({"ok": "good", "suspect": "warn", "stop": "crit"}[v], {"ok": "关口 ok", "suspect": "关口 suspect", "stop": "关口 stop"}[v]))
        f = fun.get(c["id"])
        if f and f["first_stop"] in ("rules", "vlm", "route"):
            bs.append(badge("info", {"rules": "规则拦", "vlm": "VLM 拦", "route": "路由到 v1.3"}[f["first_stop"]]))
        if c.get("flags"):
            bs.append(badge("warn", "水印"))
        return f'<div class="card"><div class="ph">{ph}</div><div class="tx"><span class="id">{esc(c["id"])}</span><span class="bs">{"".join(bs)}</span></div></div>'

    # ---- per-clip master table
    def master_row(c):
        r = v0(c["id"]) or {}
        rf = by.get(c["id"], {}).get("ref") or {}
        f = fun.get(c["id"], {})
        p = par.get(c["id"], {}).get("rot_p90_over_flow_med")
        verdict = r.get("verdict", "")
        vb = badge({"ok": "good", "suspect": "warn", "stop": "crit"}.get(verdict, "info"), verdict or "—")
        oc = r.get("outcome", "")
        ob = {"true_pass": badge("good", "通过且准"), "silent_error": badge("crit", "静默错误"), "false_stop": badge("warn", "误拦"),
              "true_stop": badge("info", "拦对了")}.get(oc, badge("info", "无参考") if r.get("ref_failed") == "True" else "")
        return [f'<span class="mono">{esc(c["id"])}</span>', esc(c.get("category") if c["role"] == "easy" else c.get("expected_failure")),
                num(p, 2) if p is not None else "", esc(f.get("rules_reason") or f.get("rules") or ""), esc(f.get("vlm_reason") or f.get("vlm") or ""),
                vb, esc((r.get("flags") or "").replace(";", " ")), num(r.get("focal_w"), 2), num(rf.get("focal_w"), 2) if rf and rf.get("n_points") else "",
                num(r.get("focal_err_pct"), 0), num(r.get("ate_vs_ref_pct"), 2), ob]

    # ---- variants table
    def variants():
        rows = []
        for run, label in (("v0a", "v0a  PINHOLE，初值 1.2×长边（复刻 GLRMap 默认）"), ("v0b", "v0b  同上，换建图种子"), ("v1sp", "v1sp  SIMPLE_PINHOLE（fx = fy）"),
                           ("v2prior", "v2prior  GeoCalib 焦距作初值，仍优化"), ("v3fixed", "v3fixed  GeoCalib 焦距固定"), ("v0a_hq960", "v0a 在 960 px（子集）")):
            rs = [by[c][run] for c in ids_e if run in by.get(c, {}) and by[c][run].get("outcome")]
            if not rs:
                continue
            acc = sum(r["silver_accurate"] == "True" for r in rs)
            rows.append([esc(label), len(rs), sum(r["verdict"] != "stop" for r in rs), acc, sum(r["outcome"] == "silent_error" for r in rs),
                         sum(r["outcome"] == "false_stop" for r in rs), num(statistics.median([abs(float(r["focal_err_pct"])) for r in rs if r.get("focal_err_pct")]), 0) + " %"])
        return table(["变体", "有银参考的片段", "过关口（ok / suspect）", "与银参考一致", "静默错误", "误拦", "焦距误差中位数"], rows) if rows else \
            '<p class="pending">变体（v0b / v1sp / v2prior / v3fixed）还没跑完，跑完后重新生成本页。</p>'

    # ---- sections
    gal_e = "".join(card(c) for c in easy)
    gal_c = "".join(card(c) for c in ctl)
    tbl_master = table(["片段", "类型", "视差指数", "规则", "VLM", "关口 v0a", "关口标志", "f/W 复刻", "f/W 银参考", "焦距差 %", "ATE %", "结局"],
                       [master_row(c) for c in easy + ctl], "tall", "视差指数 = 旋转模型残差 / 光流（纯摇镜 ≤ 0.06，有真视差 ≥ 0.8）。焦距差取 fx、fy 中较差的一个。结局只在银参考成功时给出。")
    demo_tbl = (table(["片段", "路由", "关口", "worker", "v1c", "kdis (°)", "xdepth_k", "分级", "最终", "原因"],
                      [[esc(d.get(k, "")) for k in ("clip", "route", "gate", "status", "v1c", "kdis", "xdepth_k", "tier", "final", "why")] for d in demo], "",
                      "稠密深度和证人是替身（见限制），v1.3 一侧没有标签所以没有回退：这张表说明各段接得上、关口和质检规则按预期反应，不说明标签精度。"
                      "v1c &gt; 1 就拒绝；mk_43622 的 xdepth_k 来自两次不同的稀疏点插值（替身深度对替身证人），不是标签质量的结论。")
                if demo else '<p class="pending">联调演示还没跑完。</p>')
    st = stats
    rev = st.get("review", {})
    body = f"""
<header class="mast"><div class="wrap">
<div class="eyebrow">E15 · GLR 轨道 · 沙箱报告</div>
<h1>前后端过滤 + GLRMap 打标，能放心批量跑吗</h1>
<p class="lede">把 3DData_anno 的前端筛选（传统规则 + VLM）和后端质检接到 GLRMap 打标上，用 {len(easy)} 条简单网络视频和 {len(ctl)} 条硬对照在没有 GPU 的沙箱里试跑。</p>
<ul class="facts"><li>2026-10-29</li><li>仅 CPU（4 核）</li><li>分支 <code>e15-glr-track</code></li><li>第一阶段 = pycolmap 复刻，VLM = Claude 替身</li></ul>
<p class="caution"><b>读结果前：</b>沙箱里没有 GLRMap 代码、GPU，也连不上你的服务器。第一阶段是 pycolmap 的降本复刻（640 px、60 帧、顺序匹配），比真实 GLRMap 更苛刻；VLM 判官是 Claude 不是 Qwen3-VL-32B。下面是线索和已经接通的管线，不是 GLRMap 的验收。</p>
</div></header>
<nav class="nav" aria-label="目录"><div class="wrap"><a href="#sum">结论</a><a href="#todo">三个要求</a><a href="#flow">管线</a><a href="#set">片段集</a><a href="#front">前端筛选</a><a href="#gate">关口</a><a href="#k">内参 K</a><a href="#demo">联调</a><a href="#eng">工程</a><a href="#limit">限制</a><a href="#next">下一步</a></div></nav>
<main class="wrap">

<section id="sum"><h2>结论<small>number first, then what it means</small></h2>
<div class="kpis">
<div class="kpi"><b>{len(easy)} + {len(ctl)}</b><span>条片段：简单网络视频 + 硬对照{('（' + str(n_local) + ' 条对照因许可证只在本地试过，不进 manifest）') if n_local else ''}</span></div>
<div class="kpi {'good' if f_easy else ''}"><b>{(str(fe_pass) + ' / ' + str(len(f_easy))) if f_easy else '—'}</b><span>简单片段通过前端（规则 + VLM 替身）</span></div>
<div class="kpi {'good' if f_ctl else ''}"><b>{(str(fc_caught) + ' / ' + str(len(f_ctl))) if f_ctl else '—'}</b><span>硬对照没能变成 GLR 标签（被规则、VLM、路由或关口挡住）</span></div>
<div class="kpi warn"><b>{len(e_pass)} / {len(e_rows)}</b><span>简单片段通过第一阶段出口关口（复刻 v0a）</span></div>
<div class="kpi crit"><b>{(str(n_f10) + ' / ' + str(len(fe))) if fe else '—'}</b><span>有银参考的片段里，焦距与参考相差 &gt; 10 %</span></div>
<div class="kpi good"><b>{st.get('tests', '—')}</b><span>项自动测试通过（含与原脚本逐项对拍）</span></div>
</div>
<div class="find">
<div><strong>1 · “简单”的网络视频，第一阶段并不容易过。</strong>复刻的第一阶段给出模型的 {len(e_model)} 条里，{len(e_pass)} 条过关口；其余被 <code>fx_fy_mismatch</code>（fx 与 fy 差 &gt; 10 %）、每图点数太少、平面对占比高或没出模型挡住。位姿本身大多是准的：与银参考比，{n_ate_ok} / {len(ate)} 条的 Sim(3) ATE ≤ 3.27 %。</div>
<div><strong>2 · 主要风险是内参 K，不是轨迹。</strong>自由焦距的 PINHOLE 在前进镜头、远景航拍上焦距几乎不可观测：{n_f10} / {len(fe)} 条的焦距与银参考差 &gt; 10 %，其中 {n_silent} 条还通过了关口（静默错误），v1c 的 K 容差（HFOV 13.2°，约 25 % 焦距）也不会拦。这和评估会话里的 KITTI / 无人机结论一致。</div>
<div><strong>3 · 前端和关口是互补的两层。</strong>规则 + VLM 挡掉纯摇镜、静止、夜景、切镜、人群；开车类片段 VLM 判“可重建”，由路由送去 v1.3；第一阶段关口再兜住前端漏掉的退化（{len(c_stop)} / {len(c_rows)} 条对照被它 stop）。{len(f_ctl) - len(single)} / {len(f_ctl)} 条对照被至少两层独立挡住；只靠一层的是 {', '.join(single) if single else '（无）'}（鱼眼，只有 VLM 拦住）。混入的 1 条淡水印（ia_telangana）规则和 VLM 都没看见。</div>
<div><strong>4 · 管线接通了，并且按“查不出就拒绝”设计。</strong>路由 → 作业包 → worker（关口先于 RoMa / PDA）→ 收集 → 质检分级 → 合并 / 回退 v1.3，在真实片段上贯通；任何一环失败、缺证据、缺深度都变成拒绝，不会悄悄出标签。</div>
<div><strong>5 · 没做、也做不了的。</strong>没有跑真实 GLRMap、RoMa、PDA、ViPE、Pi3X；证人质检只有接线测试。第二轮独立评审因预算停止。这些都在“下一步”里给了服务器上的执行清单。</div>
</div></section>

<section id="todo"><h2>你的三个要求，各做到哪一步<small>request → delivered → not done</small></h2>
{table(["要求", "做了什么", "没做到"], [
 ["1 · 按计划继续做", "Phase 0 里沙箱能做的部分：关口规则 + 回放工具、第一阶段复刻与消融、前端筛选在新片段上的回放；服务器上要做的部分写成带通过线 / 放弃线的 runbook。", "离线回放要的服务器数据；真实 GLRMap 联调（沙箱没有）。"],
 ["2 · 下载简单些的 OOD 视频再测", f"从 Mixkit、Pexels、Internet Archive、美国国家公园局 B-roll 找到并逐条核验了 {len(easy)} 条 DL3DV 难度的片段，另配 {len(ctl)} 条硬对照；全部跑了前端筛选、第一阶段复刻、银参考。", "YouTube 在沙箱里被拦（没有绕过）；没有许可证干净的鱼眼对照；真实 GLRMap 没跑。"],
 ["3 · 前后端过滤和打标结合", "<code>pipeline/run_batch_glr.sh</code>：原样复用 3DData_anno 的筛选与 v1.3 质检，中间插入路由、GLRMap worker（带第一阶段出口关口）、跨模型证人质检、T1/T2/T3 分级和回退 v1.3；现有脚本一个没改。", "真实 GLRMap 命令行 / 输出布局是按文档推断的，第一次在服务器上跑要照核对清单。"]])}
</section>

<section id="flow"><h2>管线<small>front filter → router → GLRMap stage 1 + gate → dense → back QC → merge</small></h2>
<ol class="steps">
<li><b>前端筛选</b>3DData_anno 原样：切镜、亮度模糊、OCR、光流、GeoCalib + VLM 判官，<code>decide.py</code> 出 pass / reject。</li>
<li class="new"><b>路由</b>只让“静态场景 + 真视差”进 GLRMap；动态、旋转为主、驾驶、具身走 v1.3。自适应抽帧 60–120 帧。</li>
<li class="new"><b>GLRMap 第一阶段</b>COLMAP SIFT + 顺序匹配 + GLOMAP 全局建图。</li>
<li class="new"><b>出口关口</b>注册率、点数、fx/fy、平面对、重投影、三角化角、K 证人分歧。stop 就不跑 RoMa / PDA，回退 v1.3。</li>
<li class="new"><b>稠密阶段</b>RoMa v2 + Pi3X 先验 + PDA 深度（SfM 尺度）。</li>
<li class="new"><b>后端质检</b>v1c 三项（对 ViPE 的轨迹、对 DA3 的深度、K 与 GeoCalib/Pi3X/DA3 的分歧）+ 旗标；查不出来一律拒绝。</li>
<li class="new"><b>分级与合并</b>T1 / T2 用 GLR 标签；T3 或失败回退 v1.3 标签（v1c 通过才算）；再不行就拒绝。</li>
</ol><p class="note">蓝框 = 本次新增；其余是 3DData_anno 现有代码。</p></section>

<section id="set"><h2>片段集<small>{len(easy)} easy + {len(ctl)} hard negative controls</small></h2>
<p>来源与许可：Mixkit（{sum(c['license'].startswith('Mixkit') for c in easy)} 条简单 + {sum(c['license'].startswith('Mixkit') for c in ctl)} 条对照，Mixkit 免费许可）、Pexels（{sum(c['license'].startswith('Pexels') for c in easy)}）、Internet Archive（{sum(c['source'].startswith('archive') for c in easy)}，CC0 / CC BY / CC BY-SA）、美国国家公园局 B-roll（{sum('NPS' in c['license'] for c in easy)}，公有领域）。每条 12 s、15 fps、不含音频；清单、sha256、许可证和下载地址在 <code>docs/experiments/E15_ood_manifest.json</code>，<code>tools/fetch_ood_set.py</code> 可按清单重建。</p>
<h3>独立核验发现的问题</h3>
<ul class="tight">
<li><b>mk_3110</b> 当初按“横移滑轨、视差最好”选入，独立核验发现是纯摇镜（旋转模型残差 0.013×光流，三角化角 1.3°），已改作无视差对照。光流里的视差代理分不开摇镜和平移，所以另外加了旋转模型视差指数：{('纯摇镜 ≤ ' + num(max(pans), 2) + '，' + str(len(t_easy_par)) + ' 条简单片段 ≥ ' + num(min(t_easy_par), 2)) if pans and t_easy_par else '待补'}。</li>
<li><b>ia_telangana</b> 带一条淡水印（不透明度约 0.2–0.3，占画面 0.5 %，全程固定），接触表分辨率下看不见；保留为“水印探针”。</li>
<li>39 条 Mixkit 记录的作者字段写成“无署名”是错的（页面标记里有贡献者链接），已按页面重取；CC0 的 ia_telangana 是上传者自述。</li>
<li>许可证干净的对照里没有<b>鱼眼</b>；{n_local} 条“仅限个人使用”的对照（含 2 条鱼眼）只在本地试过，不进 manifest，不展示画面。</li></ul>
<h3>简单片段</h3><div class="gal">{gal_e}</div>
<h3>硬对照</h3><div class="gal">{gal_c}</div>
<p class="note">缩略图是各片段第 90 帧；CC BY / CC BY-SA 的片段只列文字，避免带出署名 / 相同方式共享的义务。</p></section>

<section id="front"><h2>前端筛选：规则 + VLM 替身<small>filter/s1–s5 unchanged on CPU · Claude as a labelled surrogate for the judge</small></h2>
<p>规则阶段（切镜、亮度 / 模糊、OCR、光流、GeoCalib）用 <code>tools/filter_stage_cpu.py</code> 原样跑在 CPU 上，只换了设备。VLM 判官用两位独立的 Claude 评审，读匿名接触表，提示词和 JSON 模式与 <code>filter/vlm_run.py</code> 相同，结果标 <code>vlm_claude</code>。它不是 Qwen3-VL-32B，换成真模型前这一层的数字只作参考。</p>
{figure(a.figs, 'fig_funnel.png', '每种场景 / 失败类型里，哪一层最先挡住片段（灰色 = 全部通过）。', '各层拦截统计')}
<ul class="tight"><li>VLM 两位评审在 {stats.get('vlm_agree', '—')} 上给出相同的“可重建”结论。</li>
<li><b>OCR 阶段有个 int16 溢出</b>：文本框面积在框大于约 181×181 px 时算错，会少报大字幕（已记入 KNOWN_ISSUES 第 9 条，没有改原脚本，因为会改变已记录的分数）。另外“前进镜头 + 画面中央一排亮窗”会被当成静态文字（<code>static_text_area</code> 误报）。</li>
<li>开车类对照（3 条）VLM 判“可重建”：它们的拦截靠路由（<code>vlm_driving</code>）和第一阶段关口。</li></ul></section>

<section id="gate"><h2>第一阶段出口关口<small>pycolmap replica of GLRMap stage 1 · {len(e_rows)} easy + {len(c_rows)} controls</small></h2>
{figure(a.figs, 'fig_gate_signals.png', '每条关口规则上各片段的位置。红线 = stop 阈值，阴影一侧会被拦；黄线 = suspect。点是复刻 v0a（640 px、60 帧）的信号。', '关口信号分布')}
<p>阈值在看到任何片段之前就定下，没有为这批数据调过。复刻比真实第一阶段苛刻（分辨率低、无引导匹配），所以“每图点数”“平面对占比”两项不能直接和评估会话的数字比；因这类信号被拦的片段会在 960 px 重跑再下结论。</p>
{variants()}
{figure(a.figs, 'fig_outcomes.png', '以银参考为标准，关口做了什么：通过且准 / 拦对了 / 误拦 / 静默错误。', '关口结局计数')}
<details open><summary>逐片段总表（{len(easy) + len(ctl)} 行）</summary>{tbl_master}</details></section>

<section id="k"><h2>内参 K：焦距为什么不稳<small>the dominant failure on web clips</small></h2>
<p>银参考 = 另一次 SfM（每 6 帧、穷举匹配、SIMPLE_RADIAL、增量建图），不是真值，所以这里先问“两次独立重建的焦距是否一致”，再看 GeoCalib 站谁。</p>
{figure(a.figs, 'fig_variants_vs_reference.png', '每个简单片段对银参考：焦距误差（横）与 ATE（纵）。绿框 = 预声明的“一致”区域；标记形状 = 关口裁决。', '变体对银参考')}
<ul class="tight"><li>航拍和前进镜头里焦距几乎不可观测：复刻的焦距常停在初值 1.2×长边附近，fx 与 fy 各自漂移，被 <code>fx_fy_mismatch</code> 拦下；或者通过关口但焦距偏差 20–80 %。</li>
<li>银参考也会错（例如 mk_41574：复刻 0.81、GeoCalib 0.81、参考 1.15），所以单靠它不能判“静默错误”；这部分在 E 表里用三方对照单独列出，属于事后分析，不进预声明的结局统计。</li>
<li>修法不在门槛上：评估会话的焦距扫描（E14）、K 先验初始化、固定 K、SIMPLE_PINHOLE，需要在服务器上用真实 GLRMap 做 Phase 0.2b 的消融，runbook 里写了通过线。</li></ul></section>

<section id="demo"><h2>联调演示<small>route → worker (gate) → collect → QC → merge on real clips</small></h2>
<p>在真实片段上走完阶段 a（路由 + 作业包）→ b（worker：第一阶段复刻 + 真关口 + 稠密替身）→ c（收集、质检分级、合并）。稠密深度用稀疏点插值替身，证人用银参考重建替身，v1.3 一侧没有标签，所以没有回退标签可用。</p>
{demo_tbl}</section>

<section id="eng"><h2>工程与评审<small>what changed in the repo</small></h2>
<ul class="tight">
<li>新增（现有脚本一个没改）：<code>pipeline/{{run_batch_glr.sh, glr_prepare.py, glr_worker.sh, glr_stage1_gate.py, glr_collect.py, glr_merge.py, glr_common.py}}</code>、<code>configs/pipeline_glr_v0.yaml</code>、<code>tests/glr/</code>（{st.get('tests', '—')} 项）、<code>tools/</code> 与 <code>evolution/E15_glr/</code> 的沙箱工具。</li>
<li><b>第一轮独立评审</b>（4 个视角，{rev.get('raw', 63)} 条原始发现：高 {rev.get('high', 16)} / 中 {rev.get('medium', 33)} / 低 {rev.get('low', 14)}），逐条由另一个代理复现，已复现的全部修复并写成测试。最重的几条：关口自己崩溃会被读成 suspect（放行）；合并时旧软链接不更新；没有 GLR 作业时阶段 c 崩溃；竖屏片段的焦距规则；模板里的引号 / 注入；taskset / timeout 没套住整个命令。</li>
<li>修复过程中自己又发现并修了：参考模型选错（<code>sparse/0</code> 不一定是最大的）、fx 与 fy 只比一个、参考失败时仍给结局。</li>
<li><b>第二轮评审（检查修复本身）因预算停止，没有结果。</b>代替它的是更多的表驱动测试和变异检查。</li></ul></section>

<section id="limit"><h2>限制：这些结论不能下<small>what this does not show</small></h2>
<ul class="tight">
<li>不能说“GLRMap 在简单网络视频上通过率是 X”。复刻是降本版（640 px、60 帧、顺序匹配、无引导）；真实 GLRMap 用全分辨率、约 100 帧、GPU SIFT。</li>
<li>不能说“前端 + VLM 的召回是 X”。VLM 是 Claude 替身，片段 {len(easy) + len(ctl)} 条，样本小。</li>
<li>不能说“质检能抓住 GLR 的错”。证人质检只有接线测试，稠密深度和证人都是替身。</li>
<li>银参考是第二次 SfM，不是真值；K 的对错没有地面真值。</li></ul></section>

<section id="next"><h2>下一步<small>server-side Phase 0 and decisions</small></h2>
<p>在服务器上按 <code>docs/GLR_PHASE0_RUNBOOK.md</code> 做（每步有命令、耗时、通过线 / 放弃线，阈值在看结果前固定）：</p>
<ul class="tight"><li><b>0.2 关口离线回放</b>：用已有的第一阶段输出回放关口，对标签算捕获率 / 误拦率（<code>tools/glr_gate_replay.py</code>）。</li>
<li><b>0.2b 内参策略消融</b>：同一份匹配库上跑 PINHOLE / SIMPLE_PINHOLE / 先验初值 / 固定焦距（<code>evolution/E15_glr/run_sweep.sh</code> 的配方）。</li>
<li><b>0.3 用真实 Qwen3-VL 重跑本页的片段集</b>，对照 Claude 替身的结论。</li>
<li><b>1 第一次真实 GLRMap 联调</b>：先 <code>glr_worker.sh JOB --dry-run</code>，按 <code>docs/GLR_INTEGRATION.md</code> §6 核对命令模板和深度文件布局。</li></ul>
<h3>需要你决定</h3>
<ul class="tight"><li>K 策略要不要作为第一阶段的默认（先验初始化 / 固定 / SIMPLE_PINHOLE），还是先做 E14 焦距扫描？</li>
<li>GLR 路由是否只收“静态 + 真视差”的片段（现在的设计），还是先在 A 组有真值的片段上和 v1.3 并排跑？</li>
<li>GLRMap 仓库的访问权限：沙箱里读不到，命令模板还是推断的。</li></ul></section>
</main>
<footer><div class="wrap">代码与文档：<a href="{REPO_URL}/docs/GLR_INTEGRATION.md">GLR_INTEGRATION.md</a> · <a href="{REPO_URL}/docs/GLR_PHASE0_RUNBOOK.md">Phase 0 runbook</a> · <a href="{REPO_URL}/docs/experiments/LOG.md">LOG.md（E15）</a> · <a href="{REPO_URL}/docs/experiments/E15_ood_manifest.json">片段清单</a> · <a href="{REPO_URL}/evolution/E15_glr/README.md">沙箱脚本</a>。本页由 <code>evolution/E15_glr/build_report_page.py</code> 从结果文件生成。</div></footer>
"""
    fragment = f'<title>GLR 轨道 E15 测试</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n<style>{CSS}</style>\n{body}'
    full = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
            + fragment + "\n</body></html>\n")
    return fragment, full


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_html")
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--funnel")
    ap.add_argument("--parallax")
    ap.add_argument("--figs")
    ap.add_argument("--frames-root")
    ap.add_argument("--demo")
    ap.add_argument("--stats")
    ap.add_argument("--fragment")
    ap.add_argument("--md", help="also write a GitHub-Markdown version (figures linked as <md-fig-dir>/<png>)")
    ap.add_argument("--md-fig-dir", default="figures")
    a = ap.parse_args(argv)
    fragment, full = build(a)
    Path(a.out_html).write_text(full)
    if a.fragment:
        Path(a.fragment).write_text(fragment)
    if a.md:
        Path(a.md).write_text(html_to_md(fragment, a.md_fig_dir))
    print(f"wrote {a.out_html} ({len(full) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
