#!/usr/bin/env python3
"""
Improvement experiments for the apple-leaf manuscript (run on a GPU / Colab).

E1  augmentation control    : supervised-only with weak vs STRONG augmentation
E2  adaptive-threshold SSL  : per-class quantile threshold (fixes the 0.90 cold-start failure)
E3  field-domain SSL        : unlabelled pool = PlantVillage pool + UNLABELLED PlantDoc images
                              (gate / no gate), evaluated on a disjoint PlantDoc half
E4  few-shot field adapt.   : BN calibration on unlabelled PlantDoc + prototype / logistic probe
                              on K labelled field images per class, 600 episodes

Everything is evaluated on the SAME held-out PlantDoc half (PD_EVAL), which is never used for
training, pseudo-labelling or model selection. Labels of PD_UNL are never read by training.

USAGE (Colab):
  python improve_experiments.py --pv_dir /content/PlantVillage_apple --pd_dir /content/PlantDoc \
      --out /content/out --seeds 0 1 2 3 4 --fractions 0.05 0.10 --exps E1 E2 E3 E4

pv_dir : folders Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy
pd_dir : train/ and test/ each with 'Apple Scab Leaf', 'Apple rust leaf', 'Apple leaf'
"""
import argparse, glob, json, os, random, time
import numpy as np, torch, timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

PV_CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
PD_MAP = {"Apple Scab Leaf": 0, "Apple rust leaf": 2, "Apple leaf": 3}   # 'Apple leaf' = healthy (assumption!)
PRESENT = [0, 2, 3]
p = argparse.ArgumentParser()
p.add_argument("--pv_dir", required=True); p.add_argument("--pd_dir", required=True); p.add_argument("--out", default="out")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10])
p.add_argument("--exps", nargs="+", default=["E1", "E2", "E3", "E4"])
p.add_argument("--img", type=int, default=224); p.add_argument("--epochs", type=int, default=15); p.add_argument("--rounds", type=int, default=3)
p.add_argument("--bs", type=int, default=32); p.add_argument("--workers", type=int, default=2)
p.add_argument("--no_pretrained", action="store_true")
A = p.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(A.out, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((A.img, A.img)), norm])
weak_tf = T.Compose([T.RandomResizedCrop(A.img, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.ColorJitter(0.2, 0.2, 0.2), norm])
strong_tf = T.Compose([T.RandomResizedCrop(A.img, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.RandAugment(2, 9),
                       norm, T.RandomErasing(p=0.5)])

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- data ----------------
def list_pv():
    f, y = [], []
    for i, c in enumerate(PV_CLASSES):
        for x in sorted(glob.glob(os.path.join(A.pv_dir, c, "*"))): f.append(x); y.append(i)
    return np.array(f), np.array(y)

def list_pd():
    f, y = [], []
    for sp in ("train", "test"):
        for d, i in PD_MAP.items():
            for x in sorted(glob.glob(os.path.join(A.pd_dir, sp, d, "*"))): f.append(x); y.append(i)
    return np.array(f), np.array(y)

def strat_split(y, fr, rng):
    a, b = [], []
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0]); n = int(round(len(ids) * fr)); a += list(ids[:n]); b += list(ids[n:])
    return np.array(sorted(a)), np.array(sorted(b))

class DS(Dataset):
    def __init__(s, files, labels, tf, two=None): s.f, s.y, s.tf, s.two = files, labels, tf, two
    def __len__(s): return len(s.f)
    def __getitem__(s, i):
        im = Image.open(s.f[i]).convert("RGB")
        if s.two is not None: return s.tf(im), s.two(im), int(s.y[i])   # weak view, strong view
        return s.tf(im), int(s.y[i])

def loader(ds, shuffle, bs=None): return DataLoader(ds, batch_size=bs or A.bs, shuffle=shuffle, num_workers=A.workers, drop_last=False)

# ---------------- model (same as the paper) ----------------
class ChannelAtt(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__(); s.mlp = nn.Sequential(nn.Conv2d(ch, ch // r, 1, bias=False), nn.ReLU(), nn.Conv2d(ch // r, ch, 1, bias=False))
    def forward(s, x): return torch.sigmoid(s.mlp(x.mean((2, 3), keepdim=True)) + s.mlp(x.amax((2, 3), keepdim=True)))
class SpatialAtt(nn.Module):
    def __init__(s, k=7): super().__init__(); s.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
    def forward(s, x): return torch.sigmoid(s.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
class CBAM(nn.Module):
    def __init__(s, ch): super().__init__(); s.ca, s.sa, s.last_mask = ChannelAtt(ch), SpatialAtt(), None
    def forward(s, x):
        x = x * s.ca(x); m = s.sa(x); s.last_mask = m.detach(); return x * m
class Net(nn.Module):
    def __init__(s, cbam=True, n=4):
        super().__init__()
        try: s.backbone = timm.create_model("efficientnet_b0", pretrained=not A.no_pretrained, features_only=True)
        except Exception as e:
            print("pretrained weights unavailable -> random init:", e); s.backbone = timm.create_model("efficientnet_b0", pretrained=False, features_only=True)
        ch = s.backbone.feature_info.channels()[-1]
        s.cbam = CBAM(ch) if cbam else nn.Identity()
        s.head = nn.Sequential(nn.Dropout(.3), nn.Linear(ch, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, n))
    def feat(s, x): return s.cbam(s.backbone(x)[-1]).mean((2, 3))
    def forward(s, x): return s.head(s.feat(x))

# ---------------- training / eval ----------------
def train(model, dl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * len(dl))); sc = torch.amp.GradScaler(enabled=DEV == "cuda")
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"): loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()

@torch.no_grad()
def predict(model, files, labels=None):
    model.eval(); P, F = [], []
    for x, _ in loader(DS(files, labels if labels is not None else np.zeros(len(files), int), eval_tf), False, 64):
        x = x.to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            f = model.feat(x); P.append(torch.softmax(model.head(f).float(), 1).cpu()); F.append(f.float().cpu())
    return torch.cat(P).numpy(), torch.cat(F).numpy()

@torch.no_grad()
def attn_entropy_and_views(model, files):
    """returns prob(weak/eval view), prob(strong view), CBAM mask entropy"""
    model.eval(); P, Q, H = [], [], []
    for xw, xs, _ in loader(DS(files, np.zeros(len(files), int), eval_tf, two=strong_tf), False, 64):
        xw, xs = xw.to(DEV), xs.to(DEV)
        P.append(torch.softmax(model(xw).float(), 1).cpu())
        m = model.cbam.last_mask if isinstance(model.cbam, CBAM) else None
        if m is not None:
            m = m.flatten(1).float(); m = m / (m.sum(1, keepdim=True) + 1e-8); H.append((-(m * (m + 1e-8).log()).sum(1)).cpu())
        Q.append(torch.softmax(model(xs).float(), 1).cpu())
    return torch.cat(P).numpy(), torch.cat(Q).numpy(), (torch.cat(H).numpy() if H else np.zeros(len(files)))

def metrics(prob, y):
    pr = prob[:, PRESENT].argmax(1); yy = np.array([PRESENT.index(v) for v in y]); rec = []; f1 = []
    for i in range(3):
        tp = ((pr == i) & (yy == i)).sum(); fp = ((pr == i) & (yy != i)).sum(); fn = ((pr != i) & (yy == i)).sum()
        rec.append(tp / max(1, tp + fn)); f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return dict(acc=float((pr == yy).mean() * 100), f1_3=float(np.mean(f1) * 100), rec=[float(r * 100) for r in rec])

# ---------------- pseudo-labelling ----------------
def select(prob, mode, rnd, base_thr, qfrac, cap_per_class, gate=None):
    conf, lab = prob.max(1), prob.argmax(1); keep = np.zeros(len(conf), bool)
    for c in range(4):
        idx = np.where(lab == c)[0]
        if len(idx) == 0: continue
        if mode == "fixed": thr = base_thr + 0.02 * rnd; ok = idx[conf[idx] >= thr]
        else:                       # per-class quantile: keep the most confident qfrac of what the model calls class c
            k = max(1, int(len(idx) * min(0.9, qfrac + 0.15 * rnd))); ok = idx[np.argsort(-conf[idx])[:k]]; ok = ok[conf[ok] >= 0.5]
        if gate is not None: ok = ok[gate[ok]]
        ok = ok[np.argsort(-conf[ok])[:cap_per_class]]; keep[ok] = True
    return keep, lab

def self_train(model, lab_f, lab_y, unl_f, mode, gate_kind, seed, base_thr=0.9, qfrac=0.3, lab_tf=weak_tf):
    log = []
    for r in range(A.rounds):
        pw, ps, H = attn_entropy_and_views(model, unl_f)
        gate = None
        if gate_kind == "agree": gate = pw.argmax(1) == ps.argmax(1)
        if gate_kind == "attn": gate = (pw.argmax(1) == ps.argmax(1)) & (H <= np.median(H))
        keep, lab = select(pw, mode, r, base_thr, qfrac, int(0.8 * len(unl_f) / 4), gate)
        log.append(int(keep.sum()))
        if keep.sum() == 0: continue
        files = np.concatenate([lab_f, unl_f[keep]]); ys = np.concatenate([lab_y, lab[keep]])
        isu = np.concatenate([np.zeros(len(lab_f), bool), np.ones(keep.sum(), bool)])
        class Mixed(Dataset):
            def __len__(s): return len(files)
            def __getitem__(s, i):
                im = Image.open(files[i]).convert("RGB"); return (strong_tf if isu[i] else lab_tf)(im), int(ys[i])
        train(model, loader(Mixed(), True), A.epochs, 5e-5)
    return log

# ---------------- few-shot (E4) ----------------
def bn_calibrate(model, files):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None   # cumulative average
    model.train()
    with torch.no_grad():
        for x, _ in loader(DS(files, np.zeros(len(files), int), eval_tf), False, 64): model(x.to(DEV))
    model.eval()

def nrm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def fewshot_eval(F, y, own_prob, Ks=(1, 5, 10), n_ep=600, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed); out = {}
    idx = {c: np.where(y == c)[0] for c in PRESENT}
    for K in Ks:
        acc = {"proto": [], "probe": [], "fuse": []}; rec = {k: np.zeros(3) for k in acc}; cnt = np.zeros(3)
        for _ in range(n_ep):
            S = {c: rng.permutation(idx[c])[:K] for c in PRESENT}
            sup = np.concatenate([S[c] for c in PRESENT]); ys = np.concatenate([[i] * K for i in range(3)])
            qi = np.setdiff1d(np.arange(len(y)), sup); qi = qi[np.isin(y[qi], PRESENT)]; yq = np.array([PRESENT.index(v) for v in y[qi]])
            P = nrm(np.stack([F[S[c]].mean(0) for c in PRESENT])); z = 10 * nrm(F[qi]) @ P.T
            clf = LogisticRegression(max_iter=300, C=10).fit(nrm(F[sup]), ys)
            preds = {"proto": z.argmax(1), "probe": clf.predict(nrm(F[qi])),
                     "fuse": (0.5 * z + 0.5 * np.log(own_prob[qi][:, PRESENT] + 1e-8)).argmax(1)}
            for k, pr in preds.items():
                acc[k].append((pr == yq).mean())
                for i in range(3): rec[k][i] += ((pr == i) & (yq == i)).sum()
            for i in range(3): cnt[i] += (yq == i).sum()
        for k in acc:
            a = np.array(acc[k]); out[f"{k}_K{K}"] = dict(acc=float(a.mean() * 100), ci95=float(1.96 * a.std(ddof=1) / np.sqrt(len(a)) * 100),
                                                          rec=(rec[k] / (cnt / 1.0) * 100).tolist())
    return out

# ---------------- main ----------------
def main():
    pv_f, pv_y = list_pv(); pd_f, pd_y = list_pd()
    print(f"PV {len(pv_f)} {np.bincount(pv_y)}   PD {len(pd_f)} {np.bincount(pd_y, minlength=4)}")
    # fixed PlantDoc split: half unlabelled (labels never used), half evaluation (identical for every method and seed)
    r0 = np.random.default_rng(1234); u_i, e_i = strat_split(pd_y, 0.5, r0)
    pd_unl, pd_ev, pd_ev_y = pd_f[u_i], pd_f[e_i], pd_y[e_i]
    print(f"PD_UNL {len(u_i)} (unlabelled)   PD_EVAL {len(e_i)} {np.bincount(pd_ev_y, minlength=4)}")
    res = []
    for seed in A.seeds:
        rng = np.random.default_rng(seed)
        te_i, rest = strat_split(pv_y, 0.15, rng); va_i, pool = strat_split(pv_y[rest], 0.15 / 0.85, np.random.default_rng(seed + 1)); va_i, pool = rest[va_i], rest[pool]
        for fr in A.fractions:
            l_i, u_pv = strat_split(pv_y[pool], fr, np.random.default_rng(seed + 2)); l_i, u_pv = pool[l_i], pool[u_pv]
            lab_f, lab_y, unl_pv = pv_f[l_i], pv_y[l_i], pv_f[u_pv]
            def base_model(cbam=True, tf=weak_tf):
                seed_all(seed); m = Net(cbam).to(DEV); train(m, loader(DS(lab_f, lab_y, tf), True), A.epochs, 1e-4); return m
            def record(name, m, extra=None):
                prob, F = predict(m, pd_ev, pd_ev_y); mt = metrics(prob, pd_ev_y)
                ptest, _ = predict(m, pv_f[te_i], pv_y[te_i]); mt["pv_acc"] = float((ptest.argmax(1) == pv_y[te_i]).mean() * 100)
                mt.update(name=name, seed=seed, frac=fr, **(extra or {})); res.append(mt); print(json.dumps(mt))
                json.dump(res, open(os.path.join(A.out, "results.json"), "w"), indent=1); return prob, F
            if "E1" in A.exps:
                record("E1_sup_weakaug", base_model(True, weak_tf))
                record("E1_sup_STRONGaug", base_model(True, strong_tf))
                record("E1_sup_noCBAM_weak", base_model(False, weak_tf))
            if "E2" in A.exps:
                for mode in ("fixed", "quantile"):
                    m = base_model(True); lg = self_train(m, lab_f, lab_y, unl_pv, mode, "none", seed); record(f"E2_SSL_{mode}", m, dict(pl=lg))
            if "E3" in A.exps:
                pool_all = np.concatenate([unl_pv, pd_unl])          # PlantVillage pool + UNLABELLED field images
                for gk in ("none", "agree", "attn"):
                    m = base_model(True); lg = self_train(m, lab_f, lab_y, pool_all, "quantile", gk, seed); record(f"E3_fieldSSL_gate-{gk}", m, dict(pl=lg))
            if "E4" in A.exps:
                for tag, ssl in (("sup", False), ("ssl", True)):
                    m = base_model(True)
                    if ssl: self_train(m, lab_f, lab_y, unl_pv, "quantile", "none", seed)
                    for bn in (False, True):
                        if bn: bn_calibrate(m, pd_unl)
                        prob, F = predict(m, pd_ev, pd_ev_y); fs = fewshot_eval(F, pd_ev_y, prob, seed=seed)
                        res.append(dict(name=f"E4_{tag}_{'BNcal' if bn else 'noBN'}", seed=seed, frac=fr, own=metrics(prob, pd_ev_y), fewshot=fs))
                        print(res[-1]["name"], {k: round(v["acc"], 1) for k, v in fs.items() if "K5" in k})
                        json.dump(res, open(os.path.join(A.out, "results.json"), "w"), indent=1)
    print("done ->", os.path.join(A.out, "results.json"))

if __name__ == "__main__": main()
