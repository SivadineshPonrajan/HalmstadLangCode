"""Twin-aware dev split. Near-duplicate prompts (char-ngram cosine >= 0.75 to
any of their 5 nearest dev neighbours) are joined into groups; whole groups go
to one half. Greedy assignment balances label x script x long(>4k chars) counts. Seed 0.
Writes dev_split_group.json with keys A, B (dev ids). Dev labels are used
ONLY to balance the halves, never to train anything."""
import json, random, collections, unicodedata, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, ".")
from common2 import load

TH, K = 0.75, 5
ids, dv, y = load("dev.csv")
def script(s):
    c = collections.Counter()
    for ch in s:
        if ch.isalpha():
            try: n = unicodedata.name(ch).split()[0]
            except ValueError: continue
            if n != "LATIN": c[n] += 1
    return c.most_common(1)[0][0] if c else "LATIN"
scr = [script(t) for t in dv]

M = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=200000,
                    sublinear_tf=True).fit_transform([t[-1500:] for t in dv])
d, nb = NearestNeighbors(n_neighbors=K+1, metric="cosine").fit(M).kneighbors(M)
par = list(range(len(dv)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(dv)):
    for dist, j in zip(d[i, 1:], nb[i, 1:]):
        if 1 - dist >= TH: par[find(i)] = find(j)
groups = collections.defaultdict(list)
for i in range(len(dv)): groups[find(i)].append(i)
G = list(groups.values())

rng = random.Random(0); rng.shuffle(G); G.sort(key=len, reverse=True)   # big groups first
cnt = {h: collections.Counter() for h in "AB"}; size = {"A": 0, "B": 0}; half = {}
for g in G:
    need = collections.Counter((scr[i], y[i], len(dv[i]) > 4000) for i in g)
    def cost(h):
        o = "B" if h == "A" else "A"
        imb = sum(abs((cnt[h][k] + need[k]) - cnt[o][k]) for k in set(cnt[h]) | set(cnt[o]) | set(need))
        return imb + abs((size[h] + len(g)) - size[o])
    h = min("AB", key=lambda h: (cost(h), rng.random()))
    for i in g: half[i] = h
    cnt[h].update(need); size[h] += len(g)

A = [ids[i] for i in range(len(dv)) if half[i] == "A"]; B = [ids[i] for i in range(len(dv)) if half[i] == "B"]
json.dump({"A": A, "B": B}, open("dev_split_group.json", "w"))
yA = np.array([y[i] for i in range(len(dv)) if half[i] == "A"]); yB = np.array([y[i] for i in range(len(dv)) if half[i] == "B"])
multi = [g for g in G if len(g) > 1]
print(f"groups: {len(G)} ({len(multi)} with >1 prompt, covering {sum(map(len, multi))} prompts; largest {len(G[0])})")
print(f"A: {len(A)} rows, U {100*yA.mean():.1f}%   B: {len(B)} rows, U {100*yB.mean():.1f}%")
worst = max(abs(cnt['A'][k] - cnt['B'][k]) for k in set(cnt['A']) | set(cnt['B']))
print(f"largest script x label x long imbalance between halves: {worst} rows")
L = np.array([len(t) > 4000 for t in dv]); hA = np.array([half[i] == "A" for i in range(len(dv))])
print(f"long (>4k) prompts: A {L[hA].sum()}, B {L[~hA].sum()}")
mp = json.load(open("minpair_slice.json"))["pairs"]; pos = {i: k for k, i in enumerate(ids)}
print(f"minimal pairs split across halves: {sum(half[pos[a]] != half[pos[b]] for a, b in mp)} of {len(mp)} (old split: 118)")
