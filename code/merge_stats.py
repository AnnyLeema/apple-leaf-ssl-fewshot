"""Merge several results.json files and print the statistics used in Tables 6-8.
usage: python3 merge_stats.py out/results.json out_more/results.json out_E3/results.json [out_baselines/results.json] [--frac 0.1]"""
import sys, json, itertools, numpy as np
from scipy import stats
argv = sys.argv[1:]; frac = 0.10
if "--frac" in argv:
    i = argv.index("--frac"); frac = float(argv[i + 1]); del argv[i:i + 2]
args = argv
rows = {}
for f in args:
    for r in json.load(open(f)): rows[(r["name"], r["seed"], round(r["frac"], 3))] = r      # later files overwrite duplicates
R = [r for k, r in rows.items() if k[2] == round(frac, 3)]
names = sorted({r["name"] for r in R})
def tci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), (x.std(ddof=1) if n > 1 else 0.0), (stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")), n
def series(name, getter):
    d = {r["seed"]: getter(r) for r in R if r["name"] == name}; return d
def paired(a, b):
    s = sorted(set(a) & set(b))
    if len(s) < 3: return None
    x, y = np.array([a[i] for i in s]), np.array([b[i] for i in s]); d = x - y
    t = stats.ttest_rel(x, y); w = stats.wilcoxon(x, y) if np.any(d != 0) else None
    return dict(n=len(s), diff=d.mean(), p_t=t.pvalue, p_w=(w.pvalue if w else 1.0), dz=d.mean() / (d.std(ddof=1) + 1e-12))
print(f"=== fraction {frac}  files: {args}")
print("\n--- Field accuracy on PD_EVAL (supervised / SSL runs) ---")
acc = {}
for n in names:
    if n.startswith("E4"): continue
    d = series(n, lambda r: r["acc"]); acc[n] = d; m, sd, ci, k = tci(list(d.values()))
    f1 = tci([r["f1_3"] for r in R if r["name"] == n])[0]; pv = tci([r["pv_acc"] for r in R if r["name"] == n])[0]
    print(f"{n:32s} n={k}  acc {m:5.1f} ± {sd:4.1f}  (95% CI ±{ci:4.1f})  F1 {f1:5.1f}  PV-test {pv:5.1f}")
print("\n--- Few-shot (E4 + E5) : accuracy, mean over seeds ---")
fs_names = [n for n in names if n.startswith("E4") or n.startswith("E5")]
for n in fs_names:
    for K in (1, 5, 10):
        for h in ("proto", "probe", "fuse"):
            key = f"{h}_K{K}"
            def g(r, key=key): return (r["fewshot"][key]["acc"]) if key in r["fewshot"] else np.nan
            d = series(n, g); v = [x for x in d.values() if not np.isnan(x)]
            if v and K == 5:
                m, sd, ci, k = tci(v); print(f"{n:28s} {key:10s} n={k}  {m:5.1f} ± {sd:4.1f}  (95% CI ±{ci:4.1f})")
print("\n--- Paired comparisons (same seeds) ---")
def cmp(label, a, b):
    r = paired(a, b)
    if r: print(f"{label:60s} n={r['n']}  Δ={r['diff']:+5.1f}  p(t)={r['p_t']:.3f}  p(Wilcoxon)={r['p_w']:.3f}  d_z={r['dz']:+.2f}")
A = lambda n: acc.get(n, {})
cmp("SSL(quantile) - supervised(weak aug)", A("E2_SSL_quantile"), A("E1_sup_weakaug"))
cmp("SSL(fixed) - supervised(weak aug)", A("E2_SSL_fixed"), A("E1_sup_weakaug"))
cmp("SSL(quantile) - supervised(STRONG aug)", A("E2_SSL_quantile"), A("E1_sup_STRONGaug"))
cmp("SSL(quantile) - SSL(fixed)", A("E2_SSL_quantile"), A("E2_SSL_fixed"))
cmp("supervised weak - no CBAM", A("E1_sup_weakaug"), A("E1_sup_noCBAM_weak"))
for g in ("agree", "attn"): cmp(f"field-SSL gate-{g} - gate-none", A(f"E3_fieldSSL_gate-{g}"), A("E3_fieldSSL_gate-none"))
cmp("field-SSL gate-attn - gate-agree", A("E3_fieldSSL_gate-attn"), A("E3_fieldSSL_gate-agree"))
for b in ("fixmatch", "flexmatch"): cmp(f"SSL(quantile) - {b}", A("E2_SSL_quantile"), A(f"E5_{b}"))
for h in ("proto", "probe", "fuse"):
    g = lambda n: series(n, lambda r, h=h: r["fewshot"][f"{h}_K5"]["acc"]) if any(r["name"] == n for r in R) else {}
    cmp(f"5-shot {h}: SSL feats - supervised feats", g("E4_ssl_noBN"), g("E4_sup_noBN"))
    cmp(f"5-shot {h}: noBN - BNcal (SSL)", g("E4_ssl_noBN"), g("E4_ssl_BNcal"))
    for b in ("fixmatch", "flexmatch"): cmp(f"5-shot {h}: SSL feats - {b} feats", g("E4_ssl_noBN"), g(f"E5_{b}"))
