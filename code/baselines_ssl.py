"""
E5: published semi-supervised baselines (FixMatch, FlexMatch) under EXACTLY the same protocol as improve_experiments.py
(same seeds, same PlantVillage split, same 10 % labels, same PlantDoc PD_EVAL, same few-shot evaluation).
Put this file next to improve_experiments.py in /content and run, e.g.

!python /content/baselines_ssl.py --pv_dir /content/pvrepo/raw/color --pd_dir /content/PlantDoc \
        --out /content/drive/MyDrive/apple_project/out_baselines --seeds 0 1 2 3 4 5 6 7 --fractions 0.10 --steps 1500
"""
import sys, argparse, json, os
q = argparse.ArgumentParser()
q.add_argument("--pv_dir", required=True); q.add_argument("--pd_dir", required=True); q.add_argument("--out", default="out_baselines")
q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7]); q.add_argument("--fractions", type=float, nargs="+", default=[0.10])
q.add_argument("--methods", nargs="+", default=["fixmatch", "flexmatch"])
q.add_argument("--steps", type=int, default=1500); q.add_argument("--bs", type=int, default=32); q.add_argument("--mu", type=int, default=2)
q.add_argument("--tau", type=float, default=0.95); q.add_argument("--lam", type=float, default=1.0); q.add_argument("--workers", type=int, default=2)
q.add_argument("--img", type=int, default=224); q.add_argument("--no_pretrained", action="store_true")
B = q.parse_args()
# hand the shared options to improve_experiments (it parses sys.argv at import time)
sys.argv = [sys.argv[0], "--pv_dir", B.pv_dir, "--pd_dir", B.pd_dir, "--out", B.out, "--img", str(B.img), "--bs", str(B.bs),
            "--workers", str(B.workers)] + (["--no_pretrained"] if B.no_pretrained else [])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, copy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import improve_experiments as ie

DEV = ie.DEV

class Inf:                       # endless iterator over a DataLoader
    def __init__(s, dl): s.dl, s.it = dl, iter(dl)
    def next(s):
        try: return next(s.it)
        except StopIteration: s.it = iter(s.dl); return next(s.it)

def run(method, lab_f, lab_y, unl_f, seed):
    ie.seed_all(seed); m = ie.Net(True).to(DEV); ema = copy.deepcopy(m).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    dl_l = DataLoader(ie.DS(lab_f, lab_y, ie.weak_tf), batch_size=B.bs, shuffle=True, num_workers=B.workers, drop_last=True)
    dl_u = DataLoader(ie.DS(unl_f, np.zeros(len(unl_f), int), ie.weak_tf, two=ie.strong_tf), batch_size=B.bs * B.mu, shuffle=True,
                      num_workers=B.workers, drop_last=True)
    L, U = Inf(dl_l), Inf(dl_u)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, B.steps); sc = torch.amp.GradScaler(enabled=DEV == "cuda")
    sigma = np.zeros(4); mask_rate = []
    for step in range(B.steps):
        m.train(); xl, yl = L.next(); xw, xs, _ = U.next()
        xl, yl, xw, xs = xl.to(DEV), yl.to(DEV), xw.to(DEV), xs.to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out = m(torch.cat([xl, xw, xs])).float()
        ol, ow, os_ = out[:len(xl)], out[len(xl):len(xl) + len(xw)], out[len(xl) + len(xw):]
        ls = nn.functional.cross_entropy(ol, yl)
        with torch.no_grad():
            pw = torch.softmax(ow.detach(), 1); conf, lab = pw.max(1)
            if method == "fixmatch": thr = torch.full_like(conf, B.tau)
            else:                    # FlexMatch: class-wise curriculum threshold  T_c = tau * beta/(2-beta),  beta_c = sigma_c / max sigma
                for c in lab[conf >= B.tau].cpu().numpy(): sigma[c] += 1
                beta = sigma / max(1.0, sigma.max()); T = torch.tensor(B.tau * beta / (2 - beta), device=DEV, dtype=conf.dtype)
                thr = T[lab]
            mask = (conf >= thr).float()
        lu = (nn.functional.cross_entropy(os_, lab, reduction="none") * mask).mean()
        loss = ls + B.lam * lu
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
        with torch.no_grad():                                    # EMA weights used for evaluation, as in FixMatch
            for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
            for be, bm in zip(ema.buffers(), m.buffers()): be.copy_(bm)
        mask_rate.append(float(mask.mean()))
    return ema, float(np.mean(mask_rate[-200:]))

def main():
    pv_f, pv_y = ie.list_pv(); pd_f, pd_y = ie.list_pd()
    r0 = np.random.default_rng(1234); u_i, e_i = ie.strat_split(pd_y, 0.5, r0)
    pd_ev, pd_ev_y = pd_f[e_i], pd_y[e_i]
    print(f"PV {len(pv_f)}  PD_EVAL {len(e_i)} {np.bincount(pd_ev_y, minlength=4)}")
    os.makedirs(B.out, exist_ok=True); res = []
    for seed in B.seeds:
        rng = np.random.default_rng(seed)                      # identical splits to improve_experiments.main()
        te_i, rest = ie.strat_split(pv_y, 0.15, rng); va_i, pool = ie.strat_split(pv_y[rest], 0.15 / 0.85, np.random.default_rng(seed + 1)); va_i, pool = rest[va_i], rest[pool]
        for fr in B.fractions:
            l_i, u_pv = ie.strat_split(pv_y[pool], fr, np.random.default_rng(seed + 2)); l_i, u_pv = pool[l_i], pool[u_pv]
            lab_f, lab_y, unl_pv = pv_f[l_i], pv_y[l_i], pv_f[u_pv]
            for meth in B.methods:
                m, mr = run(meth, lab_f, lab_y, unl_pv, seed)
                prob, F = ie.predict(m, pd_ev, pd_ev_y); mt = ie.metrics(prob, pd_ev_y)
                pt, _ = ie.predict(m, pv_f[te_i], pv_y[te_i]); mt["pv_acc"] = float((pt.argmax(1) == pv_y[te_i]).mean() * 100)
                fs = ie.fewshot_eval(F, pd_ev_y, prob, seed=seed)
                mt.update(name=f"E5_{meth}", seed=seed, frac=fr, mask_rate=mr, fewshot=fs); res.append(mt)
                print(meth, "seed", seed, "acc %.1f" % mt["acc"], "pv %.1f" % mt["pv_acc"], "5-shot proto %.1f" % fs["proto_K5"]["acc"])
                json.dump(res, open(os.path.join(B.out, "results.json"), "w"), indent=1)
    print("done ->", os.path.join(B.out, "results.json"))

if __name__ == "__main__": main()
