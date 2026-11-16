"""Build every evaluation condition ONCE from dev (seeded) so all systems
see identical text. 13 SUITE conditions + many-shot with the payload moved
to the middle and to the start (never used in any training)."""
import json, os, random
import transforms as T
from common2 import load

def many_shot_at(s, rng, where):
    n = rng.randint(20, 60); items = []
    for _ in range(n):
        q, a = rng.choice(T._FILLER).split("||"); items += [q, a]
    pos = {"mid": len(items)//2, "start": 0}[where]
    items.insert(pos, s); return json.dumps(items, ensure_ascii=False)

ids, txt, _ = load("dev.csv")
os.makedirs("evalsets", exist_ok=True)
rng = random.Random(1)
for name, fn in T.SUITE.items():
    json.dump([fn(t, rng=rng) for t in txt], open(f"evalsets/{name}.json", "w"), ensure_ascii=False)
for w in ("mid", "start"):
    json.dump([many_shot_at(t, rng, w) for t in txt], open(f"evalsets/many_shot_{w}.json", "w"), ensure_ascii=False)
print("wrote", sorted(os.listdir("evalsets")))
