"""Build the enlarged training sets from train.csv ONLY (seeded, balanced).

R1 = train + 2 extra copies of every opposite-label near-twin pair member
         + real 40-shot JSON prompts oversampled to 2x the smaller class per label
R2 = R1 + one code-mix noise variant of every non-JSON train prompt
Writes train_r1.csv, train_r2.csv and build_train_report.json."""
import csv, json, random, collections, unicodedata, difflib, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, ".")
from common2 import load

TWIN_TH, TWIN_MAX_WORDS, TWIN_COPIES, SHOT_MULT = 0.75, 5, 2, 2
P_SPLIT, P_MOVE, WIN = 0.3, 0.3, 3
rng = random.Random(0)
ids, tr, y = load("train.csv")
rows = [(ids[i], tr[i], "U" if y[i] else "S") for i in range(len(tr))]
rep = {}

# ---- twin pairs (opposite-label near-duplicates inside train) ----
M = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=200000,
                    sublinear_tf=True).fit_transform([t[-1500:] for t in tr])
d, nb = NearestNeighbors(n_neighbors=2, metric="cosine").fit(M).kneighbors(M)
cand = {tuple(sorted((i, int(nb[i,1])))) for i in range(len(tr))
        if 1 - d[i,1] >= TWIN_TH and y[i] != y[nb[i,1]]}
def words_differing(a, b):
    sm = difflib.SequenceMatcher(None, tr[a].split(), tr[b].split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
# keep only TRUE minimal pairs: similar AND differing by at most TWIN_MAX_WORDS words
# (most candidate pairs share a long template but differ in >15 words)
pairs = {p for p in cand if 1 <= words_differing(*p) <= TWIN_MAX_WORDS}
rep["twin_candidates"] = len(cand)
members = sorted({k for p in pairs for k in p})
twin = [(f"{ids[k]}#tw{c}", tr[k], "U" if y[k] else "S") for k in members for c in range(TWIN_COPIES)]
rep["twin_pairs"] = len(pairs); rep["twin_members"] = len(members)

# ---- real 40-shot JSON prompts, balanced oversampling ----
def is40(t):
    try:
        v = json.loads(t.strip()); return isinstance(v, list) and len(v) == 40
    except Exception:
        return False
shot = {l: [k for k in range(len(tr)) if is40(tr[k]) and y[k] == l] for l in (0, 1)}
m = min(len(shot[0]), len(shot[1])); rep["shot40_S"], rep["shot40_U"] = len(shot[0]), len(shot[1])
extra = [(f"{ids[k]}#ms{c}", tr[k], "U" if l else "S")
         for l in (0, 1) for c, k in enumerate(rng.choices(shot[l], k=SHOT_MULT * m))]

R1 = rows + twin + extra

# ---- code-mix noise (friend's augmentation; may change meaning) ----
def nonlatin(tok):
    for ch in tok:
        if ch.isalpha():
            try:
                if unicodedata.name(ch).split()[0] != "LATIN": return True
            except ValueError:
                pass
    return False
def codemix(s):
    toks = s.split()
    out = []
    for t in toks:                                   # 1) split some non-Latin words
        # split only at character-cluster boundaries: never before a combining
        # mark (category M*, e.g. Indic vowel signs / virama), so no orphan marks
        cuts = [c for c in range(1, len(t)) if unicodedata.category(t[c])[0] != "M"] if nonlatin(t) else []
        if len(t) >= 4 and cuts and rng.random() < P_SPLIT:
            c = rng.choice(cuts); out += [t[:c], t[c:]]
        else:
            out.append(t)
    for i in range(len(out)):                        # 2) move some next to a Latin neighbour
        if nonlatin(out[i]) and rng.random() < P_MOVE:
            j = i + rng.choice((-1, 1))
            if 0 <= j < len(out) and not nonlatin(out[j]): out[i], out[j] = out[j], out[i]
    idx = [i for i, t in enumerate(out) if nonlatin(t)]   # 3) shuffle non-Latin words in windows
    for w in range(0, len(idx), WIN):
        win = idx[w:w+WIN]; vals = [out[i] for i in win]; rng.shuffle(vals)
        for i, v in zip(win, vals): out[i] = v
    return " ".join(out)
cm = [(f"{ids[k]}#cm", codemix(tr[k]), "U" if y[k] else "S") for k in range(len(tr)) if not is40(tr[k])]
R2 = R1 + cm

def script_of(s):
    c = collections.Counter()
    for ch in s:
        if ch.isalpha():
            try: n = unicodedata.name(ch).split()[0]
            except ValueError: continue
            if n != "LATIN": c[n] += 1
    return c.most_common(1)[0][0] if c else "LATIN"
def write(name, R):
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "prompt", "label"]); w.writerows(R)
    c = collections.Counter(r[2] for r in R)
    per = collections.defaultdict(collections.Counter)
    for r in R: per[script_of(r[1])][r[2]] += 1
    rep[name] = {"rows": len(R), "S": c["S"], "U": c["U"], "U_pct": round(100 * c["U"] / len(R), 2),
                 "U_pct_per_script": {k: round(100 * v["U"] / (v["S"] + v["U"]), 1) for k, v in sorted(per.items())}}
write("train_r1.csv", R1); write("train_r2.csv", R2)
rep["codemix_variants"] = len(cm); rep["codemix_changed"] = sum(r[1] != tr[k] for r, k in zip(cm, [k for k in range(len(tr)) if not is40(tr[k])]))
json.dump(rep, open("build_train_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
