"""Qwen3Guard (Gen-4B from round 2 on) LoRA classifier fine-tuning, binary S/U.

Head = linear layer on the final hidden state at the LAST templated token
(the generation-prompt position, same position E4's last-token probe read).
Trains on train.csv ONLY. Checkpoint and threshold are selected on dev-A;
dev-B, the minimal-pair slice and the eval sets are only ever scored.

  python3 12_sft.py --tag s0 --limit_train 200 --limit_eval 200 --epochs 1   # smoke
  python3 12_sft.py --tag s0                                                # main
  python3 12_sft.py --tag jumble --jumble 0.5                               # ablation
"""
import argparse, glob, json, math, os, random, time
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from model_inputs import InputBuilder, SYSTEM_PROMPT
from common2 import load, J, best_threshold, dev_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="local Qwen3Guard-Gen-4B dir")
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--max_tokens", type=int, default=32768)
ap.add_argument("--tokens_per_batch", type=int, default=16384)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--evals_per_epoch", type=int, default=2)
ap.add_argument("--jumble", type=float, default=0.0, help="prob. of word-shuffling a training prompt")
ap.add_argument("--jumble_window", type=int, default=3, help="window for local word shuffling; 0 = global")
ap.add_argument("--many_shot_aug", type=float, default=0.0, help="prob. per epoch of wrapping a TRAINING prompt in 20-60 benign Q/A shots, real prompt at a random position")
ap.add_argument("--system", action="store_true", help="prepend the pre-registered SYSTEM_PROMPT (ablation)")
ap.add_argument("--limit_train", type=int, default=0)
ap.add_argument("--limit_eval", type=int, default=0)
ap.add_argument("--predict_only", action="store_true")
ap.add_argument("--score_train", action="store_true", help="with --predict_only: score the ORIGINAL train.csv prompts with the best checkpoint, save scores_train_<tag>.npz, exit")
ap.add_argument("--train_csv", default="train.csv", help="training CSV (built from train.csv only)")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
OUT = f"sft_{a.tag}"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/log.txt", "a")
def P(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
P("args", vars(a))

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
build = InputBuilder(tok, a.max_tokens, system=SYSTEM_PROMPT if a.system else None)

base = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
head = nn.Linear(base.config.hidden_size, 1).to("cuda", torch.float32)
dev = torch.device("cuda")

def jumble(s, rng):
    """LOCAL shuffle: words are shuffled only inside consecutive windows of
    --jumble_window words, so clause-level order (e.g. where an ownership or
    negation phrase attaches) is mostly kept. --jumble_window 0 = global shuffle.
    Either way this may change meaning; it is an ablation, not a default."""
    w = s.split()
    if len(w) < 4: return s
    k = a.jumble_window
    if k <= 0:
        rng.shuffle(w); return " ".join(w)
    out = []
    for i in range(0, len(w), k):
        chunk = w[i:i+k]; rng.shuffle(chunk); out += chunk
    return " ".join(out)

# disjoint from transforms._FILLER, which is used only to build eval sets
TRAIN_FILLER = [
    ("Which planet is closest to the sun?", "Mercury."),
    ("How many continents are there?", "Seven."),
    ("What is the chemical symbol for gold?", "Au."),
    ("Who wrote Hamlet?", "William Shakespeare."),
    ("How do I reverse a string in JavaScript?", "Use split, reverse and join."),
    ("What is the freezing point of water?", "0 degrees Celsius."),
    ("What does CPU stand for?", "Central Processing Unit."),
    ("How many minutes are in an hour?", "Sixty."),
]
def many_shot_train(s, rng):
    n = rng.randint(20, 60); items = []
    for _ in range(n):
        q, ans = rng.choice(TRAIN_FILLER); items += [q, ans]
    items.insert(rng.randint(0, len(items)), s)
    return json.dumps(items, ensure_ascii=False)

def encode(texts, rng=None, p_jumble=0.0):
    out = []
    for t in texts:
        # augmentations apply to TRAINING only (rng is None for every eval call)
        if rng is not None and a.many_shot_aug and rng.random() < a.many_shot_aug: t = many_shot_train(t, rng)
        if p_jumble and rng.random() < p_jumble: t = jumble(t, rng)
        out.append(build(t)[0])
    return out

def batches(seqs, shuffle, rng):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    bs, cur, mx = [], [], 0
    for i in order:
        L = len(seqs[i])
        if cur and max(mx, L) * (len(cur) + 1) > a.tokens_per_batch:
            bs.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, L)
    if cur: bs.append(cur)
    if shuffle: rng.shuffle(bs)
    return bs

def forward(seqs, idx):
    L = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        s = seqs[i]; ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    h = model(input_ids=ids, attention_mask=att).last_hidden_state
    last = h[torch.arange(len(idx), device=dev), att.sum(1) - 1]
    return head(last.float()).squeeze(-1)

@torch.no_grad()
def score(seqs):
    model.eval(); out = np.zeros(len(seqs), np.float32)
    for b in batches(seqs, False, None):
        out[b] = forward(seqs, b).float().cpu().numpy()
    model.train(); return out

# ---------------- data ----------------
_, tr, ytr = load(a.train_csv)
dids, dv, ydv = load("dev.csv")
A, B = dev_split(dids)
if a.limit_train:
    sel = np.random.default_rng(0).permutation(len(tr))[:a.limit_train]
    tr, ytr = [tr[i] for i in sel], ytr[sel]
iA = np.where(A)[0]
if a.limit_eval: iA = iA[:a.limit_eval]
seqA = encode([dv[i] for i in iA]); yA = ydv[iA]

def save(tagdir):
    model.save_pretrained(tagdir); torch.save(head.state_dict(), f"{tagdir}/head.pt")

# ---------------- train ----------------
if not a.predict_only:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.0)
    rng = random.Random(a.seed)
    seq_tr0 = encode(tr); trunc_train = base_trunc = build.n_truncated
    # Build EVERY epoch's sequences and batches up front, so the LR schedule
    # and the eval points use the true number of optimizer steps per epoch
    # (jumbling changes token lengths, hence batch counts).
    plan = []
    for ep in range(a.epochs):
        seq = encode(tr, rng, a.jumble) if (a.jumble or a.many_shot_aug) else seq_tr0
        bl = batches(seq, True, rng); plan.append((seq, bl, math.ceil(len(bl) / a.accum)))
    if a.jumble or a.many_shot_aug: trunc_train = build.n_truncated - base_trunc   # summed over the epochs actually trained on
    total = sum(p[2] for p in plan); warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/warm) * max(0.0, (total - s) / max(1, total - warm)))
    lossf = nn.BCEWithLogitsLoss(); yt = torch.tensor(ytr, dtype=torch.float32)
    P(f"train n={len(tr)} steps per epoch={[p[2] for p in plan]} total={total} "
      f"train prompts middle-truncated={trunc_train}")
    best, step, t0 = -1e9, 0, time.time()
    loss_sum, loss_n = 0.0, 0
    tl, tok_since, steps_since = time.time(), 0, 0   # for per-log step time / tokens-per-second
    model.train()   # from_pretrained returns eval mode: without this, grad checkpointing and LoRA dropout are OFF until the first eval

    def do_eval(ep):
        global best
        sA = score(seqA); th = best_threshold(yA, sA); jA = J(yA, (sA > th).astype(int))[0]
        P(f"EVAL ep {ep} step {step}: dev-A J={jA:.2f} (thr {th:.3f})")
        if jA > best:
            best = jA; save(f"{OUT}/best")
            json.dump({"step": step, "epoch": ep, "devA_J": jA, "thr": th}, open(f"{OUT}/best/meta.json", "w"))
            P("  new best -> saved")

    for ep, (seq_tr, bl, n_steps) in enumerate(plan):
        # eval points inside this epoch, the last one always at the epoch's final step
        eval_at = {max(1, round(n_steps * (k+1) / a.evals_per_epoch)) for k in range(a.evals_per_epoch)}
        local = 0
        for k, b in enumerate(bl):
            tok_since += len(b) * max(len(seq_tr[i]) for i in b)   # padded tokens, as processed
            loss = lossf(forward(seq_tr, b), yt[b].to(dev)) / a.accum
            loss.backward(); loss_sum += loss.item() * a.accum; loss_n += 1
            if (k + 1) % a.accum == 0 or k == len(bl) - 1:
                torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1; local += 1
                steps_since += 1
                if step % 10 == 0:
                    torch.cuda.synchronize(); dt = time.time() - tl
                    P(f"ep {ep} step {step}/{total} loss(mean of last {loss_n} batches) {loss_sum/loss_n:.4f} {time.time()-t0:.0f}s "
                      f"| {dt/steps_since:.2f} s/step {tok_since/dt:.0f} tok/s")
                    tl, tok_since, steps_since, loss_sum, loss_n = time.time(), 0, 0, 0.0, 0
                if local in eval_at:
                    do_eval(ep); tl, tok_since, steps_since, loss_sum, loss_n = time.time(), 0, 0, 0.0, 0
        assert local == n_steps, f"epoch {ep}: ran {local} steps, planned {n_steps}"
    P(f"training done. best dev-A J={best:.2f}. train prompts middle-truncated: {trunc_train}")

# ---------------- predict with best checkpoint ----------------
SMOKE = bool(a.limit_train or a.limit_eval)
NS = a.limit_eval or 200   # in smoke mode, predict on a small slice only
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
sd = load_file(f"{OUT}/best/adapter_model.safetensors")
set_peft_model_state_dict(model, sd); head.load_state_dict(torch.load(f"{OUT}/best/head.pt"))
meta = json.load(open(f"{OUT}/best/meta.json"))
if a.score_train:
    assert a.predict_only, "--score_train needs --predict_only"
    _, tr0, ytr0 = load("train.csv")           # original prompts, no augmentation
    st = score(encode(tr0)); np.savez(f"scores_train_{a.tag}.npz", train=st)
    P(f"wrote scores_train_{a.tag}.npz ({len(st)} train prompts); middle-truncated prompts total: {build.n_truncated}"); raise SystemExit(0)
dev_rows = np.arange(len(dv))[:NS] if SMOKE else np.arange(len(dv))
sd_ = np.full(len(dv), np.nan, np.float32); sd_[dev_rows] = score(encode([dv[i] for i in dev_rows]))
out = {"dev": sd_, "thr_devA": np.array(meta["thr"])}
for f in sorted(glob.glob("evalsets/*.json")):
    c = os.path.basename(f)[:-5]; xs = json.load(open(f))
    s = np.full(len(xs), np.nan, np.float32)
    rows = np.where(B)[0]                      # robustness scored on dev-B rows only
    if SMOKE: rows = rows[:20]
    s[rows] = score(encode([xs[i] for i in rows])); out["eval_" + c] = s
    P(f"scored {c}")
if os.path.exists("devtest.csv") and not SMOKE:
    _, dt, _ = load("devtest.csv", labelled=False); out["devtest"] = score(encode(dt))
dest = f"{OUT}/smoke_scores.npz" if SMOKE else f"scores_sft_{a.tag}.npz"   # smoke never reaches the report
np.savez(dest, **out)
P(f"wrote {dest}; middle-truncated prompts total: {build.n_truncated}")
