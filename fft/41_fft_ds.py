"""Full fine-tune (FFT) of Qwen3Guard-Gen-4B as a binary S/U classifier, DeepSpeed ZeRO-2, 4 GPUs.

Same recipe as 12_sft.py (R2) except LoRA -> ALL weights trainable (incl. the tied token embedding):
  * same InputBuilder, many-shot aug, token-budget batches, seed, batch order, 3% warmup + linear decay,
    2 epochs, 2 dev-A evals per epoch, clip 1.0, weight decay 0, head = linear on last-token state.
  * each optimizer step = `accum` (2) token-budget batches, exactly as in R2; each batch's rows are split
    over world/accum ranks (loss scaled so the summed gradient equals R2's batch-mean gradient / accum).
  * lr 1e-5 for the backbone (head 1e-3), a single fixed value, no sweep.
  * precision: fp32 master weights/optimizer states (ZeRO-2), bf16 autocast compute, head in fp32.
Trains on --train_csv ONLY (built from train.csv). Checkpoint/threshold selection on dev-A only.

  python -m torch.distributed.run --standalone --nproc_per_node=4 41_fft_ds.py --model $MODEL --tag fft ...
"""
import argparse, glob, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer, AutoModel
from model_inputs import InputBuilder
from common2 import load, J, best_threshold, dev_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--max_tokens", type=int, default=32768); ap.add_argument("--tokens_per_batch", type=int, default=16384)
ap.add_argument("--accum", type=int, default=2); ap.add_argument("--evals_per_epoch", type=int, default=2)
ap.add_argument("--many_shot_aug", type=float, default=0.0)
ap.add_argument("--limit_train", type=int, default=0); ap.add_argument("--limit_eval", type=int, default=0)
ap.add_argument("--predict_only", action="store_true")
ap.add_argument("--train_csv", default="train.csv")
a = ap.parse_args()

deepspeed.init_distributed()
rank, world = dist.get_rank(), dist.get_world_size()
lrank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lrank); dev = torch.device("cuda", lrank)
assert world % a.accum == 0, "world size must be a multiple of --accum"
G = world // a.accum                                  # ranks sharing one batch

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
OUT = f"sft_{a.tag}"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/log.txt", "a") if rank == 0 else None
def P(*x):
    if rank: return
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
P("args", vars(a), "world", world, "ranks per batch", G)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
build = InputBuilder(tok, a.max_tokens)

# ---- tie check: the original checkpoint's lm_head must equal the input embedding (tie_word_embeddings) ----
if rank == 0:
    from safetensors import safe_open
    idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]; cfgj = json.load(open(f"{a.model}/config.json"))
    t = {}
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if k in idx:
            with safe_open(f"{a.model}/{idx[k]}", "pt") as f: t[k] = f.get_tensor(k)
    P("tie_word_embeddings in config:", cfgj.get("tie_word_embeddings"), "| lm_head.weight stored:", "lm_head.weight" in t,
      "| equal to embed_tokens:", (torch.equal(t["lm_head.weight"], t["model.embed_tokens.weight"]) if len(t) == 2 else "n/a (not stored; tied)"))
    del t

backbone = AutoModel.from_pretrained(a.model, dtype=torch.float32, device_map={"": lrank})
assert backbone.config.tie_word_embeddings, "config must keep tie_word_embeddings=True"
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
for p_ in backbone.parameters(): p_.requires_grad_(True)       # FFT: every weight incl. embed_tokens
head = nn.Linear(backbone.config.hidden_size, 1).to(dev, torch.float32)

class Net(nn.Module):
    def __init__(s, b, h): super().__init__(); s.backbone, s.head = b, h
    def forward(s, ids, att):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = s.backbone(input_ids=ids, attention_mask=att).last_hidden_state
        last = h[torch.arange(ids.shape[0], device=ids.device), att.sum(1) - 1]
        return s.head(last.float()).squeeze(-1)
net = Net(backbone, head)
ntrain = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
P(f"trainable params {ntrain/1e6:.1f}M (embedding {backbone.embed_tokens.weight.numel()/1e6:.1f}M, requires_grad={backbone.embed_tokens.weight.requires_grad})")

TRAIN_FILLER = [
    ("Which planet is closest to the sun?", "Mercury."), ("How many continents are there?", "Seven."),
    ("What is the chemical symbol for gold?", "Au."), ("Who wrote Hamlet?", "William Shakespeare."),
    ("How do I reverse a string in JavaScript?", "Use split, reverse and join."),
    ("What is the freezing point of water?", "0 degrees Celsius."), ("What does CPU stand for?", "Central Processing Unit."),
    ("How many minutes are in an hour?", "Sixty."),
]
def many_shot_train(s, rng):
    n = rng.randint(20, 60); items = []
    for _ in range(n):
        q, ans = rng.choice(TRAIN_FILLER); items += [q, ans]
    items.insert(rng.randint(0, len(items)), s)
    return json.dumps(items, ensure_ascii=False)
def encode(texts, rng=None):
    out = []
    for t in texts:
        if rng is not None and a.many_shot_aug and rng.random() < a.many_shot_aug: t = many_shot_train(t, rng)
        out.append(build(t)[0])
    return out
def batches(seqs, shuffle, rng):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); bs, cur, mx = [], [], 0
    for i in order:
        L = len(seqs[i])
        if cur and max(mx, L) * (len(cur) + 1) > a.tokens_per_batch: bs.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, L)
    if cur: bs.append(cur)
    if shuffle: rng.shuffle(bs)
    return bs
def tensors(seqs, idx):
    L = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx): s = seqs[i]; ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
    return ids.to(dev), att.to(dev)

@torch.no_grad()
def score(seqs):
    net.eval(); out = torch.zeros(len(seqs), device=dev)
    for n, b in enumerate(batches(seqs, False, None)):
        if n % world == rank: out[b] = net(*tensors(seqs, b)).float()
    dist.all_reduce(out); net.train(); return out.cpu().numpy()   # each index written by exactly one rank

# ---------------- data ----------------
_, tr, ytr = load(a.train_csv); dids, dv, ydv = load("dev.csv"); A, B = dev_split(dids)
if a.limit_train:
    sel = np.random.default_rng(0).permutation(len(tr))[:a.limit_train]; tr, ytr = [tr[i] for i in sel], ytr[sel]
iA = np.where(A)[0]
if a.limit_eval: iA = iA[:a.limit_eval]
seqA = encode([dv[i] for i in iA]); yA = ydv[iA]

def save(d):
    if rank == 0:
        os.makedirs(d, exist_ok=True); backbone.save_pretrained(d); torch.save(head.state_dict(), f"{d}/head.pt")
    dist.barrier()

# ---------------- train ----------------
if not a.predict_only:
    opt = torch.optim.AdamW([{"params": [p_ for p_ in backbone.parameters()], "lr": a.lr},
                             {"params": list(head.parameters()), "lr": a.head_lr}], weight_decay=0.0)
    rng = random.Random(a.seed)
    seq_tr0 = encode(tr); base_trunc = build.n_truncated; plan = []
    for ep in range(a.epochs):
        seq = encode(tr, rng) if a.many_shot_aug else seq_tr0
        bl = batches(seq, True, rng); plan.append((seq, bl, math.ceil(len(bl) / a.accum)))
    trunc_train = (build.n_truncated - base_trunc) if a.many_shot_aug else base_trunc
    total = sum(p[2] for p in plan); warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/warm) * max(0.0, (total - s) / max(1, total - warm)))
    ds_cfg = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "gradient_clipping": 1.0,
              "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                    "reduce_bucket_size": 2e8, "allgather_bucket_size": 2e8},
              "zero_allow_untested_optimizer": True, "steps_per_print": 10**9, "wall_clock_breakdown": False}
    engine, _, _, sched = deepspeed.initialize(model=net, optimizer=opt, lr_scheduler=sched, config=ds_cfg)
    yt = torch.tensor(ytr, dtype=torch.float32)
    P(f"train n={len(tr)} steps per epoch={[p[2] for p in plan]} total={total} train prompts middle-truncated={trunc_train}")
    best, step, t0 = -1e9, 0, time.time(); loss_sum = torch.zeros((), device=dev); loss_n = 0
    tl, tok_since, steps_since = time.time(), 0, 0
    engine.train()

    def do_eval(ep):
        global best
        sA = score(seqA); th = best_threshold(yA, sA); jA = J(yA, (sA > th).astype(int))[0]
        P(f"EVAL ep {ep} step {step}: dev-A J={jA:.2f} (thr {th:.3f})")
        if jA > best:                                   # identical on every rank (same scores)
            best = jA; save(f"{OUT}/best")
            if rank == 0: json.dump({"step": step, "epoch": ep, "devA_J": float(jA), "thr": float(th)}, open(f"{OUT}/best/meta.json", "w"))
            P("  new best -> saved")

    for ep, (seq_tr, bl, n_steps) in enumerate(plan):
        eval_at = {max(1, round(n_steps * (k+1) / a.evals_per_epoch)) for k in range(a.evals_per_epoch)}; local = 0
        for s0 in range(0, len(bl), a.accum):
            grp = bl[s0:s0 + a.accum]
            j, part = rank // G, rank % G                 # which batch of the group, which slice of its rows
            rows = np.array_split(np.array(grp[j]), G)[part].tolist() if j < len(grp) else []
            for b in grp: tok_since += len(b) * max(len(seq_tr[i]) for i in b)
            if rows:
                lg = net(*tensors(seq_tr, rows))
                bce = nn.functional.binary_cross_entropy_with_logits(lg, yt[rows].to(dev), reduction="sum")
                full = len(grp[j])
                loss = bce / full / a.accum * world        # DeepSpeed averages grads over ranks -> summed grad == R2's
                loss_sum += bce.detach() / full
            else:                                          # keep collectives in step; contributes zero gradient
                lg = net(*tensors(seq_tr, [grp[0][0]])); loss = lg.sum() * 0.0
            engine.backward(loss); engine.step(); loss_n += len(grp); step += 1; local += 1; steps_since += 1
            if step % 10 == 0:
                ls = loss_sum.clone(); dist.all_reduce(ls); torch.cuda.synchronize(); dt = time.time() - tl
                P(f"ep {ep} step {step}/{total} loss(mean of last {loss_n} batches) {ls.item()/loss_n:.4f} {time.time()-t0:.0f}s "
                  f"| {dt/steps_since:.2f} s/step {tok_since/dt:.0f} tok/s | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
                tl, tok_since, steps_since, loss_n = time.time(), 0, 0, 0; loss_sum.zero_()
            if local in eval_at:
                do_eval(ep); tl, tok_since, steps_since, loss_n = time.time(), 0, 0, 0; loss_sum.zero_()
        assert local == n_steps, f"epoch {ep}: ran {local} steps, planned {n_steps}"
    P(f"training done. best dev-A J={best:.2f}. train prompts middle-truncated: {trunc_train}")
    del engine, opt; torch.cuda.empty_cache()

# ---------------- predict with best checkpoint ----------------
from safetensors.torch import load_file
SMOKE = bool(a.limit_train or a.limit_eval); NS = a.limit_eval or 200
sd = {}
for f in sorted(glob.glob(f"{OUT}/best/model*.safetensors")): sd.update(load_file(f, device=f"cuda:{lrank}"))
miss = backbone.load_state_dict(sd, strict=True); del sd
head.load_state_dict(torch.load(f"{OUT}/best/head.pt", map_location=dev)); meta = json.load(open(f"{OUT}/best/meta.json"))
dev_rows = np.arange(len(dv))[:NS] if SMOKE else np.arange(len(dv))
sd_ = np.full(len(dv), np.nan, np.float32); sd_[dev_rows] = score(encode([dv[i] for i in dev_rows]))
out = {"dev": sd_, "thr_devA": np.array(meta["thr"])}
for f in sorted(glob.glob("evalsets/*.json")):
    c = os.path.basename(f)[:-5]; xs = json.load(open(f)); s = np.full(len(xs), np.nan, np.float32)
    rows = np.where(B)[0]
    if SMOKE: rows = rows[:20]
    s[rows] = score(encode([xs[i] for i in rows])); out["eval_" + c] = s; P(f"scored {c}")
if os.path.exists("devtest.csv") and not SMOKE:
    _, dt_, _ = load("devtest.csv", labelled=False); out["devtest"] = score(encode(dt_))
dest = f"{OUT}/smoke_scores.npz" if SMOKE else f"scores_sft_{a.tag}.npz"
if rank == 0: np.savez(dest, **out)
P(f"wrote {dest}; middle-truncated prompts total: {build.n_truncated}")
dist.barrier()
