"""Devtest submissions for r1 and r2 (separate zips). No training, no selection on devtest:
threshold = best_threshold(y[dev-A], dev scores[dev-A]) on dev_split_group.json; devtest is only predicted."""
import sys, os, csv, zipfile, hashlib, numpy as np
sys.path.insert(0, ".")
from common2 import load, J, best_threshold, dev_split
ids, dv, y = load("dev.csv"); A, B = dev_split(ids)
dt_ids, _, _ = load("devtest.csv", labelled=False)
assert len(dt_ids) == 10000 and len(set(dt_ids)) == 10000
P = {}
for m in ("r1", "r2"):
    z = np.load(f"scores_sft_{m}.npz"); s = z["dev"]; sd = z["devtest"]
    assert len(sd) == 10000 and not np.isnan(sd).any() and not np.isnan(s).any()
    t = best_threshold(y[A], s[A]); jb = J(y[B], (s[B] > t).astype(int))[0]; ja = J(y[A], (s[A] > t).astype(int))[0]
    lab = np.where(sd > t, "U", "S"); P[m] = lab
    d = f"submission_{m}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["id", "label"])
        for i, l in zip(dt_ids, lab): w.writerow([i, l])
    zi = zipfile.ZipInfo("submission.csv", date_time=(2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
    with zipfile.ZipFile(f"{d}/submission.zip", "w") as zf: zf.writestr(zi, open(f"{d}/submission.csv", "rb").read())
    print(f"{m}: threshold {t:.4f} | dev-A J {ja:.2f} | dev-B J {jb:.2f} | devtest U {int((lab=='U').sum())} ({100*(lab=='U').mean():.2f}%), S {int((lab=='S').sum())} | zip sha256 {hashlib.sha256(open(f'{d}/submission.zip','rb').read()).hexdigest()[:16]}")
print(f"r1 vs r2 label agreement on devtest: {int((P['r1']==P['r2']).sum())}/10000")
