#!/usr/bin/env python3
"""stage-1 链的本地只读看板：读 cloud_pull/（watcher 的落点），生成一个自包含的 ops/dashboard.html。

约束（用户，2026-09-19）：只显示已记录的数字与预注册的阈值线；不新增指标、不产生任何判定——
门 / kill / 收敛 / 残留 / warm 的判定以 run_matrix / train.py 的输出为准，这里只是把同样的数摆出来。
不 import 项目代码、不碰 pod、不写 cloud_pull/。

用法：
  python3 ops/dashboard.py                 # 生成一次
  python3 ops/dashboard.py --watch 120     # 每 120 s 重新生成（页面自带 120 s 自动刷新）
"""
import argparse, html, json, pathlib, re, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY, LAM, MAX_STEPS, EVAL_EVERY = "ord_n8_h4_d5", 0.5, 400, 50
CHAIN = [("A1", f"A_{LAM}_{KEY}_s1", "rl"), ("B_warm-SFT(A1)", f"Bwarm_sft_{KEY}_s1", "sft"), ("B1", f"B_{LAM}_{KEY}_s1", "rl")]
STAGE1_LINE, BUDGET = 240.0, 300.0   # stage-1 内部线：用户 2026-09-20 由 $200 放宽到 $240（$200 漏算了 eval）；$300 = 预注册硬上限


def jl(p):
    if not p.exists(): return []
    out = []
    for line in p.read_text().splitlines():
        try: out.append(json.loads(line))
        except Exception: pass
    return out


def jf(p):
    try: return json.loads(p.read_text())
    except Exception: return None


def collect(pull, rate):
    now = time.time()
    adm = jf(pull / "results" / f"admission_{KEY}.json") or jf(ROOT / "results" / f"admission_{KEY}.json") or {}
    kahn = (adm.get("kahn_lb") or {}).get("2.5"); dec = (adm.get("decision_lb") or {}).get("2.5")
    runs = []
    for label, name, kind in CHAIN:
        rd = pull / "runs" / name
        steps, evals = jl(rd / "steps.jsonl"), jl(rd / "eval.jsonl")
        files = [f for f in (rd / "steps.jsonl", rd / "eval.jsonl") if f.exists()]
        mtime = max((f.stat().st_mtime for f in files), default=None)
        ckpt, unint = jf(rd / "designated_ckpt.json"), jf(rd / "uninterpretable.json")
        status = "未开始" if not (rd.exists() and (steps or evals or (rd / "designated_ckpt.json").exists())) else ("不可解释（残留检查触发）" if unint else (f"已结束：{ckpt.get('reason')} @ step {ckpt.get('step')}" if ckpt else "进行中"))
        sec = sum(s.get("sec") or 0 for s in steps) + 60 * sum(e.get("eval_min") or 0 for e in evals)
        runs.append(dict(label=label, name=name, kind=kind, exists=rd.exists(), status=status, done=bool(ckpt or unint),
                         steps=steps, evals=evals, step=(steps[-1]["step"] if steps else 0), mtime=mtime,
                         age_min=(None if mtime is None else (now - mtime) / 60), sec=sec))
    # 花费：只统计 steps.jsonl 的每步计时 + eval.jsonl 的 eval 计时（不含启动 / 权重加载 / 空转 → 是下界）
    spent_h = sum(r["sec"] for r in runs) / 3600
    rl = [r for r in runs if r["kind"] == "rl"]
    all_steps = [s["sec"] for r in rl for s in r["steps"] if s.get("sec")]
    all_ev = [e["eval_min"] for r in rl for e in r["evals"] if e.get("eval_min")]
    step_sec = (sum(all_steps) / len(all_steps)) if all_steps else 494.0
    ev_min = (sum(all_ev) / len(all_ev)) if all_ev else 60.0
    remain_h = 0.0
    for r in rl:
        if r["done"]: continue
        left = MAX_STEPS - r["step"]
        ev_left = MAX_STEPS // EVAL_EVERY + 1 - len(r["evals"])
        remain_h += (left * step_sec + max(ev_left, 0) * ev_min * 60) / 3600
    # 退出码：pod 上每个排查 / 验收 run 由启动脚本写 logs/<tag>.rc（"rc=N"，或多行 "<label> rc=N"）；watcher 拉到 cloud_pull/logs/
    rcs = {}
    for f in sorted((pull / "logs").glob("*.rc")):
        for line in f.read_text().splitlines():
            m = re.match(r"^(?:(\S+)\s+)?rc=(-?\d+|stopped\w*)", line.strip())
            if m: rcs[f.stem + ("_" + m.group(1) if m.group(1) else "")] = dict(rc=(int(m.group(2)) if m.group(2).lstrip("-").isdigit() else "stopped"), mtime=f.stat().st_mtime)
    def rc_for(name):
        hit = [k for k in rcs if name.endswith("_" + k) or name == k]
        return (max(hit, key=len), rcs[max(hit, key=len)]) if hit else (None, None)
    chain_names = {n for _, n, _ in CHAIN}; others = []; used = set()   # 链以外的 run（排查 / 验收用的 dry-run）
    rdirs = [d for d in (pull / "runs").glob("*") if d.is_dir()] if (pull / "runs").exists() else []
    for rd in rdirs:
        if rd.name in chain_names or rd.name.startswith("smoke"): continue
        st = jl(rd / "steps.jsonl"); k, rc = rc_for(rd.name); used.add(k)
        mt = max([(rd / "steps.jsonl").stat().st_mtime] if (rd / "steps.jsonl").exists() else [] + ([rc["mtime"]] if rc else []), default=0)
        if not st and not rc: continue
        others.append(dict(name=rd.name, steps=st, evals=jl(rd / "eval.jsonl"), mtime=mt, rc=(rc["rc"] if rc else None)))
    for k, rc in rcs.items():                                            # 有退出码但没有任何已完成步的 run（第 1 步之前就崩了）
        if k not in used and not any(o["name"].endswith("_" + k) for o in others):
            others.append(dict(name=k, steps=[], evals=[], mtime=rc["mtime"], rc=rc["rc"]))
    for f in (pull / "logs").glob("*.start"):                            # 启动标记：有 .start、还没有 .rc、也还没完成任何一步 → 进行中
        if f.stem not in rcs and not any(o["name"].endswith("_" + f.stem) for o in others):
            others.append(dict(name=f.stem, steps=[], evals=[], mtime=f.stat().st_mtime, rc=None, started=True))
    for o in others:
        o["age_min"] = (now - o["mtime"]) / 60
        o["state"] = "stopped" if o["rc"] == "stopped" else "crashed" if (o["rc"] not in (None, 0)) else ("done" if o["rc"] == 0 else ("running" if (o["age_min"] < 30 or (o.get("started") and o["age_min"] < 90)) else "unknown"))
    others.sort(key=lambda o: o["mtime"], reverse=True)
    for r in runs:                                                       # 链上的 run：matrix_halt.json 点名 → 崩
        hj = jf(pull / "results" / "matrix_halt.json")
        r["crashed"] = bool(hj and r["name"] in str(hj.get("reason", "")) and not r["steps"])
    agg = jf(pull / "runs" / f"B_{LAM}_{KEY}_s1" / "agg10.json") or {}                # B1 每 10 步聚合（pod 上 ops/analysis/b_agg.py 产出）
    halt, done = jf(pull / "results" / "matrix_halt.json"), (pull / "results" / "CHAIN_DONE").exists()
    warm = jf(pull / "results" / f"warm_decision_{KEY}.json")
    mlog = pull / "results" / "matrix_log.md"
    return dict(generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"), rate=rate, runs=runs, kahn2=(2 * kahn if kahn else None), kill=(1.5 * dec if dec else None),
                frozen_direct=adm.get("direct"), M=adm.get("M"), spent_h=spent_h, remain_h=remain_h, step_sec=step_sec, ev_min=ev_min,
                step_sec_measured=bool(all_steps), ev_min_measured=bool(all_ev), halt=halt, chain_done=done, warm=warm,
                agg=agg, others=others, matrix_log=(mlog.read_text().splitlines()[-12:] if mlog.exists() else []), stage1_line=STAGE1_LINE, budget=BUDGET)


CSS = """
:root{--bg:#f4f3f0;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e3e2dd;--a:#2a78d6;--b:#eb6834;--thr:#52514e;
--bad:#c4312f;--badbg:#fbe9e8;--ok:#0a6b0a;--okbg:#e6f3e6;--warnbg:#fdf3d9;--warn:#7a5200}
@media (prefers-color-scheme:dark){:root{--bg:#111110;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8a8984;--rule:#30302e;--a:#3987e5;--b:#d95926;--thr:#c3c2b7;
--bad:#f08a88;--badbg:#3a1c1b;--ok:#7fd07f;--okbg:#16301a;--warnbg:#33290f;--warn:#e9c46a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}h1{font-size:20px;margin:0 0 2px}h2{font-size:16px;margin:32px 0 4px}
.sub{color:var(--ink2);font-size:13px;margin:0 0 12px}.note{color:var(--ink2);font-size:12px;margin:6px 0 0}
.banner{padding:10px 14px;border-radius:8px;margin:12px 0;font-weight:600}.banner.bad{background:var(--badbg);color:var(--bad)}.banner.ok{background:var(--okbg);color:var(--ok)}.banner.warn{background:var(--warnbg);color:var(--warn)}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}.node{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}
.node.cur{border-color:var(--ink);border-width:2px}.node .t{font-weight:600}.node .s{color:var(--ink2);font-size:12px}
.bar{height:6px;background:var(--rule);border-radius:3px;margin-top:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ink2);border-radius:3px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.tile{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.tile .d{color:var(--ink2);font-size:12px}.tile.bad .v{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px}.card h3{font-size:13px;margin:0 0 2px}.card .cs{color:var(--ink2);font-size:12px;margin:0 0 6px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:0 0 4px;flex-wrap:wrap}.legend i{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.legend i.dash{border-top-style:dashed;border-color:var(--thr)}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{font:11px system-ui,sans-serif;fill:var(--ink2)}.empty{color:var(--muted);font-size:13px;padding:40px 0;text-align:center}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--rule)}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
.scroll{overflow-x:auto}pre{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px;overflow-x:auto;font-size:12px;white-space:pre-wrap}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;margin-right:4px}.pill.bad{background:var(--badbg);color:var(--bad)}.pill.ok{background:var(--okbg);color:var(--ok)}.pill.warn{background:var(--warnbg);color:var(--warn)}.pill.run{background:var(--rule);color:var(--ink)}
.card.run-bad{border-color:var(--bad);border-width:2px}details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13px;padding:6px 0}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:4px 8px;border-radius:4px;font-size:12px;display:none;z-index:9;white-space:nowrap}
@media (max-width:760px){.chain,.tiles{grid-template-columns:repeat(2,1fr)}.grid,.grid3{grid-template-columns:1fr}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const COL = {A1:'var(--a)', B1:'var(--b)', 'exact':'var(--b)', 'Hamming≤2':'var(--a)', 'hard 违规':'var(--bad)', '零梯度组':'var(--bad)', '全错组':'var(--ink2)'};
const tip = document.getElementById('tip');
function fmt(v,k){ if(v==null) return '–'; return k==='pct' ? (100*v).toFixed(1)+'%' : (Math.abs(v)>=100 ? Math.round(v).toLocaleString() : v.toFixed(2)); }
function roll(pts,w){ return pts.map((p,i)=>{ const s=pts.slice(Math.max(0,i-w+1),i+1); return [p[0], s.reduce((a,q)=>a+q[1],0)/s.length]; }); }
// series: [{name, pts:[[x,y]], raw:bool}]  lines: [{y,label}]  一张图一个 y 轴
function chart(el, series, o){
  series = series.filter(s=>s.pts.length);
  if(!series.length){ el.innerHTML='<div class="empty">还没有数据</div>'; return; }
  const W=520,H=230,m={l:48,r:14,t:10,b:26}, iw=W-m.l-m.r, ih=H-m.t-m.b;
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])), ys=series.flatMap(s=>s.pts.map(p=>p[1])).concat((o.lines||[]).map(l=>l.y));
  const x0=0, x1=Math.max(o.xmax||0, ...xs, 1);
  let y0=o.ymin!=null?o.ymin:Math.min(...ys), y1=o.ymax!=null?o.ymax:Math.max(...ys); if(y1===y0) y1=y0+1;
  if(o.ymin==null) y0=Math.max(0,y0-(y1-y0)*0.08); if(o.ymax==null) y1=y1+(y1-y0)*0.08;
  const X=x=>m.l+(x-x0)/(x1-x0)*iw, Y=y=>m.t+ih-(y-y0)/(y1-y0)*ih;
  let g='';
  for(let i=0;i<=4;i++){ const y=y0+(y1-y0)*i/4; g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--rule)"/><text x="${m.l-6}" y="${Y(y)+4}" text-anchor="end">${fmt(y,o.kind)}</text>`; }
  for(let i=0;i<=4;i++){ const x=Math.round(x0+(x1-x0)*i/4); g+=`<text x="${X(x)}" y="${H-6}" text-anchor="middle">${x}</text>`; }
  (o.lines||[]).forEach((l,i)=>{ g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(l.y)}" y2="${Y(l.y)}" stroke="var(--thr)" stroke-width="1.5" stroke-dasharray="5 4"/><text x="${W-m.r}" y="${Y(l.y)-5}" text-anchor="end" style="fill:var(--ink)">${l.label}</text>`; });
  series.forEach(s=>{ const c=COL[s.name];
    if(s.raw){ s.pts.forEach(p=>{ g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="1.6" fill="${c}" opacity=".35"/>`; });
      const r=roll(s.pts,o.win||10); g+=`<path d="${r.map((p,i)=>(i?'L':'M')+X(p[0])+' '+Y(p[1])).join('')}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`; }
    else { g+=`<path d="${s.pts.map((p,i)=>(i?'L':'M')+X(p[0])+' '+Y(p[1])).join('')}" fill="none" stroke="${c}" stroke-width="2"/>`;
      s.pts.forEach(p=>{ g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${c}" stroke="var(--surface)" stroke-width="2"/>`; }); }
    const last=s.pts[s.pts.length-1]; g+=`<text x="${Math.min(X(last[0])+6,W-m.r-16)}" y="${Y(last[1])-7}" style="fill:var(--ink);font-weight:600">${s.name}</text>`; });
  g+=`<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${m.t+ih}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/><rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.title||''}">${g}</svg>`;
  const svg=el.querySelector('svg'), cross=svg.querySelector('.cross');
  svg.addEventListener('mousemove',ev=>{ const r=svg.getBoundingClientRect(), px=(ev.clientX-r.left)/r.width*W, xv=x0+(px-m.l)/iw*(x1-x0);
    const rows=series.map(s=>{ let b=s.pts[0]; s.pts.forEach(p=>{ if(Math.abs(p[0]-xv)<Math.abs(b[0]-xv)) b=p; }); return {n:s.name,p:b}; });
    const near=rows.reduce((a,b)=>Math.abs(b.p[0]-xv)<Math.abs(a.p[0]-xv)?b:a);
    cross.setAttribute('x1',X(near.p[0])); cross.setAttribute('x2',X(near.p[0])); cross.setAttribute('visibility','visible');
    tip.innerHTML=rows.filter(q=>q.p[0]===near.p[0]).map(q=>`${q.n} · step ${q.p[0]} · ${fmt(q.p[1],o.kind)}`).join('<br>'); tip.style.display='block';
    tip.style.left=Math.min(ev.clientX+12,window.innerWidth-220)+'px'; tip.style.top=(ev.clientY+12)+'px'; });
  svg.addEventListener('mouseleave',()=>{ tip.style.display='none'; cross.setAttribute('visibility','hidden'); });
}
const R = Object.fromEntries(D.runs.map(r=>[r.label,r]));
const stepS = k => ['A1','B1'].map(n=>({name:n, raw:true, pts:R[n].steps.filter(s=>s[k]!=null).map(s=>[s.step,s[k]])}));
const evalS = (k,names) => (names||['A1','B1']).map(n=>({name:n, pts:R[n].evals.filter(e=>e[k]!=null).map(e=>[e.step,e[k]])}));
const a0 = R.A1.evals.find(e=>e.step===0);
const AG = (D.agg && D.agg.bins) ? D.agg.bins : [];
const agS = (k,name) => ({name, pts: AG.map(b=>[b.step, b[k]])});
if (AG.length) {
  chart(document.getElementById('b_hit'),  [agS('hit_exact','exact'), agS('hit_hamming2','Hamming≤2')], {kind:'pct', ymin:0, ymax:1});
  chart(document.getElementById('b_hard'), [agS('hard_rate','hard 违规')], {kind:'pct', ymin:0, ymax:1});
  chart(document.getElementById('b_zero'), [agS('zero_grad_frac','零梯度组'), agS('all_wrong_frac','全错组')], {kind:'pct', ymin:0, ymax:1});
} else { for (const id of ['b_hit','b_hard','b_zero']) document.getElementById(id).innerHTML='<div class="empty">还没有 agg10.json（B1 未开始或尚未拉回）</div>'; }
chart(document.getElementById('c_L'),   stepS('L_mean'),      {xmax:400, title:'训练采样 L_mean'});
chart(document.getElementById('c_acc'), stepS('reward_acc'),  {xmax:400, kind:'pct', ymin:0, ymax:1});
chart(document.getElementById('c_nat'), stepS('natural_end'), {xmax:400, kind:'pct', ymin:0, ymax:1});
chart(document.getElementById('e_L'),   evalS('L_mean'),  {xmax:400, ymin:0, lines:[...(a0?[{y:0.7*a0.L_mean,label:'操纵门：A1 step-0 L_mean 的 −30% = '+Math.round(0.7*a0.L_mean)}]:[])]});
chart(document.getElementById('e_Lm'),  evalS('L_median'),{xmax:400, ymin:0, lines:[...(D.kahn2?[{y:D.kahn2,label:'2×Kahn = '+D.kahn2+'（kill 线 '+D.kill+' 在其下方）'}]:[])]});
chart(document.getElementById('e_acc'), evalS('acc'),     {xmax:400, kind:'pct', ymin:0, ymax:1, lines:[...(a0?[{y:a0.acc-0.05,label:'操纵门：A1 step-0 acc − 5pp = '+fmt(a0.acc-0.05,'pct')}]:[])]});
const aBest = R.A1.evals.length ? Math.max(...R.A1.evals.map(e=>e.acc)) : null;
chart(document.getElementById('w_acc'), evalS('acc'),     {xmax:400, kind:'pct', ymin:0, ymax:1, lines:[...(aBest!=null?[{y:aBest-0.15,label:'warm 线：A1 最佳 acc − 15pp = '+fmt(aBest-0.15,'pct')+'（只看 B1 ≤200 步）'}]:[])]});
"""


def esc(x): return html.escape(str(x))


def conv_table(r):
    ev = r["evals"]
    if len(ev) < 2: return '<div class="empty">需要至少两次 eval</div>'
    rows, streak = "", 0
    for p, c in zip(ev, ev[1:]):
        dl = abs(c["L_mean"] - p["L_mean"]) / max(c["L_mean"], 1); da = abs(c["acc"] - p["acc"])
        small = dl < 0.05 and da < 0.04; streak = streak + 1 if small else 0
        rows += f"<tr><td>{p['step']} → {c['step']}</td><td>{100*dl:.1f}%</td><td>{100*da:.1f} pp</td><td>{'是' if small else '否'}</td><td>{streak} / 2</td></tr>"
    return f'<div class="scroll"><table><tr><th>区间</th><th>|ΔL_mean| / L_后（&lt;5%）</th><th>|Δacc|（&lt;4pp）</th><th>两项都小</th><th>连续小区间</th></tr>{rows}</table></div>'


def resid_table(r, fd):
    ev = [e for e in r["evals"] if e.get("direct_acc") is not None]
    if not ev: return '<div class="empty">还没有 direct-check 数据</div>'
    rows = ""
    for e in ev:
        thr = e.get("residual_threshold"); active = fd is not None and e["acc"] >= fd + 0.10
        rows += f"<tr><td>{e['step']}</td><td>{100*e['acc']:.1f}%</td><td>{100*e['direct_acc']:.2f}%</td><td>{'–' if thr is None else f'{100*thr:.1f}%'}</td><td>{'是' if active else '否'}</td><td>{'是' if (thr is not None and active and e['direct_acc'] >= thr) else '否'}</td></tr>"
    return f'<div class="scroll"><table><tr><th>step</th><th>带轨迹 acc</th><th>direct acc</th><th>触发线 acc − (acc − 冻结 direct)/2</th><th>本次 acc ≥ 冻结 direct + 10pp</th><th>direct ≥ 触发线</th></tr>{rows}</table></div>'


def eval_table(r):
    if not r["evals"]: return ""
    cols = [("step", "step", "{}"), ("acc", "acc", "{:.3f}"), ("L_mean", "L_mean", "{:.0f}"), ("L_median", "L_median", "{:.0f}"), ("capped_rate", "到顶率", "{:.2f}"),
            ("viol_rate", "违规率", "{:.3f}"), ("direct_acc", "direct", "{:.4f}"), ("letter_frac", "字母占比", "{:.3f}"), ("eval_min", "eval 分钟", "{:.0f}")]
    head = "".join(f"<th>{h}</th>" for _, h, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{f.format(e[k]) if e.get(k) is not None else '–'}</td>" for k, _, f in cols) + "</tr>" for e in r["evals"])
    return f'<h3 style="font-size:13px;margin:14px 0 4px">{esc(r["label"])} 的 eval 记录（表格视图）</h3><div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


STATE = dict(stopped=("人工停止", "warn"), crashed=("崩溃", "bad"), done=("完成", "ok"), running=("进行中", "run"), unknown=("状态不明（无退出码、30 分钟没更新）", "warn"))


def one_run(o):
    lab, cls = STATE[o["state"]]
    rows = "".join(f"<tr><td>{s.get('step')}</td><td>{s.get('sec', 0):.0f}</td><td>{s.get('gen_sec', 0):.0f}</td><td>{s.get('train_sec', 0):.0f}</td>"
                   f"<td>{'–' if s.get('reward_acc') is None else format(s['reward_acc'], '.2f')}</td><td>{'–' if s.get('L_mean') is None else format(s['L_mean'], '.0f')}</td>"
                   f"<td>{'–' if s.get('natural_end') is None else format(s['natural_end'], '.2f')}</td><td>{s.get('peak_reserved_gib', '–')}</td></tr>" for s in o["steps"])
    ev = "；".join(f"eval@{e.get('step')}: acc {e.get('acc')}, L_mean {e.get('L_mean')}, n {e.get('n')}" for e in o["evals"])
    rc = "" if o["rc"] is None else f" · 退出码 {o['rc']}"
    body = (f'<div class="scroll"><table><tr><th>step</th><th>秒</th><th>生成秒</th><th>训练秒</th><th>reward_acc</th><th>L_mean</th><th>自然结束率</th><th>峰值 reserved GiB</th></tr>{rows}</table></div>'
            if rows else '<p class="cs">没有任何已完成的训练步。</p>')
    return (f'<div class="card run-{cls}" style="margin-bottom:12px"><h3><span class="pill {cls}">{lab}</span> {esc(o["name"])}</h3>'
            f'<p class="cs">已完成 {len(o["steps"])} 步{rc} · {o["age_min"]:.0f} 分钟前更新{(" · " + esc(ev)) if ev else ""}</p>{body}</div>')


def others_html(d):
    if not d["others"]: return '<div class="empty">cloud_pull/ 里没有链以外的 run</div>'
    running = [o for o in d["others"] if o["state"] == "running"]
    rest = [o for o in d["others"] if o["state"] != "running"]
    top = running + rest[:1]; hist = rest[1:]
    out = "".join(one_run(o) for o in top)
    if hist:
        line = "".join(f'<tr><td><span class="pill {STATE[o["state"]][1]}">{STATE[o["state"]][0]}</span></td><td style="text-align:left">{esc(o["name"])}</td><td>{len(o["steps"])}</td><td>{"–" if o["rc"] is None else o["rc"]}</td><td>{o["age_min"] / 60:.1f} h 前</td></tr>' for o in hist)
        n_bad = sum(1 for o in hist if o["state"] == "crashed")
        out += (f'<details><summary>历史 {len(hist)} 个 run（其中崩溃 {n_bad} 个）</summary><div class="scroll"><table><tr><th>状态</th><th style="text-align:left">run</th><th>完成步数</th><th>退出码</th><th>更新</th></tr>{line}</table></div>'
                + "".join(one_run(o) for o in hist) + '</details>')
    return out


def render(d):
    runs = d["runs"]; rate = d["rate"]
    cur = next((r for r in runs if r["exists"] and not r["done"]), None)
    banners = ""
    if d["halt"]: banners += f'<div class="banner bad">⛔ 链已停（--strict）：{esc(d["halt"].get("reason"))} · {esc(d["halt"].get("time"))} — 需要用户决定</div>'
    if d["chain_done"]: banners += '<div class="banner ok">✓ CHAIN_DONE：stage-1 链已结束</div>'
    if not any(r["exists"] for r in runs): banners += '<div class="banner warn">⚠ cloud_pull/ 里还没有 stage-1 任何 run 的文件：链没开始，或 watcher 还没把 steps.jsonl 拉回来</div>'
    elif cur and cur["age_min"] is not None and cur["age_min"] > 25 and not d["halt"] and not d["chain_done"]:
        banners += f'<div class="banner bad">⚠ {esc(cur["label"])} 的本地文件已 {cur["age_min"]:.0f} 分钟没更新（正常约 8 分钟一步 + 10 分钟一次 pull）：可能在跑 eval（约 1 小时）、链停了、或 pull 断了</div>'
    nodes = ""
    for r in runs:
        pct = 100 * r["step"] / MAX_STEPS if r["kind"] == "rl" else (100 if r["done"] else 0)
        prog = f"step {r['step']} / {MAX_STEPS} · eval {len(r['evals'])} 次" if r["kind"] == "rl" else "SFT + eval-only"
        st_html = '<span class="pill bad">崩溃</span> matrix_halt 点名，未完成任何训练步' if r.get("crashed") else esc(r["status"])
        nodes += f'<div class="node{" cur" if r is cur else ""}"{' style="border-color:var(--bad);border-width:2px"' if r.get("crashed") else ""}><div class="t">{esc(r["label"])}</div><div class="s">{st_html}</div><div class="s">{prog}</div><div class="bar"><i style="width:{pct:.0f}%"></i></div></div>'
    w = d["warm"]
    nodes += f'<div class="node"><div class="t">warm 判定</div><div class="s">{"未到" if not w else esc(json.dumps(w, ensure_ascii=False))}</div></div>'
    spent, worst = d["spent_h"] * rate, (d["spent_h"] + d["remain_h"]) * rate
    age = "–" if not cur or cur["age_min"] is None else f"{cur['age_min']:.0f} 分钟前"
    tiles = (f'<div class="tile"><div class="k">当前</div><div class="v">{esc(cur["label"]) if cur else "–"}</div><div class="d">{("step %d / %d" % (cur["step"], MAX_STEPS)) if cur else "无进行中的 run"}</div></div>'
             f'<div class="tile"><div class="k">本地数据最后更新</div><div class="v">{age}</div><div class="d">本地文件时间（= pull 时间，不是 pod 时间）</div></div>'
             f'<div class="tile"><div class="k">已花（计时下界）</div><div class="v">${spent:.0f}</div><div class="d">{d["spent_h"]:.1f} GPU·h × ${rate}/h；不含启动 / 空转</div></div>'
             f'<div class="tile{" bad" if worst > d["stage1_line"] else ""}"><div class="k">stage-1 最坏投影（各跑满 400 步）</div><div class="v">${worst:.0f}</div><div class="d">对 ${d["stage1_line"]:.0f} 线 / ${d["budget"]:.0f} 上限 · 每步 {d["step_sec"]:.0f} s（{"实测" if d["step_sec_measured"] else "dry-run 值"}）· eval {d["ev_min"]:.0f} min（{"实测" if d["ev_min_measured"] else "估计"}）</div></div>')
    a1, b1 = runs[0], runs[2]
    focus = cur if cur and cur["kind"] == "rl" else (b1 if b1["exists"] else a1)
    leg = '<div class="legend"><span><i style="border-color:var(--a)"></i>A1（可用字母）</span><span><i style="border-color:var(--b)"></i>B1（禁字母）</span><span><i class="dash"></i>预注册阈值</span></div>'
    data = json.dumps(dict(runs=[dict(label=r["label"], steps=r["steps"], evals=r["evals"]) for r in runs], kahn2=d["kahn2"], kill=d["kill"], agg=d["agg"]), ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="120">
<title>letter-tax stage-1 看板</title><style>{CSS}</style></head><body><main>
<h1>letter-tax · stage-1 链看板</h1><p class="sub">生成于 {esc(d["generated"])} · 每 120 s 自动刷新 · 数据来自本地 cloud_pull/ · 只显示已记录的数字与预注册阈值线，<b>不产生任何判定</b>（判定以 run_matrix 输出为准）</p>
{banners}
<h2>1 · 现在在哪</h2><div class="chain">{nodes}</div><div class="tiles">{tiles}</div>
<p class="note">费率按 ${rate}/h 判（用户 2026-09-19 定；pod 实际单价 $1.59/h）。"已花"只累加每步与 eval 的计时，是下界；实际账单以 RunPod 为准。</p>

<h2>2 · 在下降吗（每步，训练采样）</h2><p class="sub">点 = 每步 32 条 rollout 的均值；线 = 10 步滑动平均。训练采样 T = 1.0，噪声大，只看趋势；判定用的是下面的 eval。</p>{leg}
<div class="grid3"><div class="card"><h3>L_mean（token）</h3><p class="cs">越低 = 轨迹越短</p><div id="c_L"></div></div>
<div class="card"><h3>reward_acc</h3><p class="cs">本步 rollout 的正确率</p><div id="c_acc"></div></div>
<div class="card"><h3>自然结束率</h3><p class="cs">没撞 cap 10240 的比例</p><div id="c_nat"></div></div></div>
<h3 style="font-size:14px;margin:16px 0 4px">B1 · 每 10 步一个点（训练 rollouts；pod 上 `ops/analysis/b_agg.py` 聚合）</h3>
<p class="sub">命中率 = 训练采样的 exact / Hamming≤2（与 eval.jsonl 的 hit_* 同口径，但是 T=1.0 的训练样本）；零梯度组 = 组内 16 条 r 全相等（Dr. GRPO 优势恒 0）；全错组 = 组内无一答对。描述量，不进判定。</p>
<div class="grid3"><div class="card"><h3>命中率</h3><p class="cs">exact 与 Hamming≤2</p><div id="b_hit"></div></div>
<div class="card"><h3>hard 违规率</h3><p class="cs">逃逸屏蔽（</think> 后有文本 / 第二个 think 标签），r = −5</p><div id="b_hard"></div></div>
<div class="card"><h3>无梯度组</h3><p class="cs">零奖励差组占比；全错组占比作参照</p><div id="b_zero"></div></div></div>

<h2>3 · 收敛了吗、门那几个数到哪了（每 50 步 eval，stopping 500 题，T = 0.6）</h2>{leg}
<div class="grid3"><div class="card"><h3>eval L_mean</h3><p class="cs">操纵门第 1 项：相对 A1 step-0 减少 ≥ 30%</p><div id="e_L"></div></div>
<div class="card"><h3>eval acc</h3><p class="cs">操纵门第 2 项：acc 损失 ≤ 5pp</p><div id="e_acc"></div></div>
<div class="card"><h3>eval L_median</h3><p class="cs">第 3 项：收敛时 ≥ 2×Kahn；kill：≤ 1.5×决策下限</p><div id="e_Lm"></div></div></div>
<div class="grid" style="margin-top:12px"><div class="card"><h3>停止判据逐区间（{esc(focus["label"])}）</h3><p class="cs">连续两个区间两项都小 → 收敛停止；否则 400 步</p>{conv_table(focus)}</div>
<div class="card"><h3>残留检查（{esc(focus["label"])}）</h3><p class="cs">冻结 direct = {d["frozen_direct"]}；生效条件锁存、连续两次触发 → 停 run（锁存状态以 train.py 为准）</p>{resid_table(focus, d["frozen_direct"])}</div></div>

<h2>4 · B1 对 A1（warm 判定用的数）</h2><p class="sub">预注册：B1 在 ≤ 200 步的最佳 eval acc &lt; A1 最佳 − 15pp → cold-start failure → B_warm-RL 进入待跑。</p>{leg}
<div class="grid"><div class="card"><h3>eval acc：A1 与 B1</h3><p class="cs">x = 各自的训练步</p><div id="w_acc"></div></div>
<div class="card"><h3>matrix_log.md 末尾</h3><p class="cs">run_matrix 自己写的记录（门 / warm 判定的正式输出在这里）</p><pre>{esc(chr(10).join(d["matrix_log"])) or "（还没有）"}</pre></div></div>

<h2>5 · 排查 / 验收 run（链以外的 dry-run）</h2><p class="sub">不是实验数据。默认只展开在跑的和最近一个，其余折叠；状态来自 pod 上启动脚本写的退出码（logs/*.rc），崩溃标红。</p>{others_html(d)}

<h2>表格视图</h2>{eval_table(a1)}{eval_table(b1)}
<div id="tip"></div><script type="application/json" id="data">{data}</script><script>{JS}</script></main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-dir", default=str(ROOT / "cloud_pull")); ap.add_argument("--out", default=str(ROOT / "ops" / "dashboard.html"))
    ap.add_argument("--usd-per-hour", type=float, default=1.6); ap.add_argument("--watch", type=int, default=0, help="每 N 秒重新生成")
    a = ap.parse_args()
    while True:
        pathlib.Path(a.out).write_text(render(collect(pathlib.Path(a.pull_dir), a.usd_per_hour)), encoding="utf-8")
        print(time.strftime("%H:%M:%S"), "wrote", a.out, flush=True)
        if not a.watch: break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
