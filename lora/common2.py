"""Shared helpers for the SFT round. Train on train.csv ONLY, always."""
import csv, json, os, numpy as np
csv.field_size_limit(10**9)

def load(path, labelled=True):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = [r["id"] for r in rows]; txt = [r["prompt"] for r in rows]
    y = np.array([1 if r["label"] == "U" else 0 for r in rows]) if labelled else None
    return ids, txt, y

def J(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    tp = ((yt==1)&(yp==1)).sum(); fn = ((yt==1)&(yp==0)).sum()
    tn = ((yt==0)&(yp==0)).sum(); fp = ((yt==0)&(yp==1)).sum()
    ru = 100*tp/max(tp+fn,1); rs = 100*tn/max(tn+fp,1)
    return ru+rs-100, ru, rs

def best_threshold(y, s):
    """Threshold maximising J. Call ONLY with dev-A scores."""
    c = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 400)))
    js = [J(y, (s > t).astype(int))[0] for t in c]
    return float(c[int(np.argmax(js))])

def paired_bootstrap(y, pa, pb, n=2000, seed=0):
    """CI for J(b) - J(a) on the SAME prompts."""
    rng = np.random.default_rng(seed); y, pa, pb = map(np.asarray, (y, pa, pb))
    d = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(y), len(y))
        d[i] = J(y[ix], pb[ix])[0] - J(y[ix], pa[ix])[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

def dev_split(dev_ids, path="dev_split.json"):
    """Return boolean masks (devA, devB) over dev rows.
    MUST reuse the split from earlier rounds so numbers stay comparable.
    Adapt the key names below to the existing file if they differ."""
    sp = json.load(open(path))
    keys = {k.lower().replace("-", "").replace("_", ""): k for k in sp}
    ka = next((keys[k] for k in ("deva", "a", "selection") if k in keys), None)
    kb = next((keys[k] for k in ("devb", "b", "report") if k in keys), None)
    if ka is None or kb is None:
        raise KeyError(f"dev_split.json keys not recognised: {list(sp)} - adapt dev_split()")
    pos = {i: n for n, i in enumerate(dev_ids)}
    def mask(v):
        m = np.zeros(len(dev_ids), bool)
        for x in v: m[x if isinstance(x, int) else pos[x]] = True
        return m
    A, B = mask(sp[ka]), mask(sp[kb])
    assert not (A & B).any() and (A | B).all(), "dev split must partition dev"
    return A, B

def minpair_mask(dev_ids, path="minpair_slice.json"):
    s = set(json.load(open(path))["ids"])
    return np.array([i in s for i in dev_ids])
