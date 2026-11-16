"""Toy check: 4 ranks, accum=2 batches/step, each batch split over G=2 ranks, loss scaled as in 41_fft_ds.py.
Compare DeepSpeed's global grad norm (pre-clip) and the resulting weight update with the R2-style single-process step."""
import os, numpy as np, torch, torch.nn as nn, torch.distributed as dist, deepspeed
deepspeed.init_distributed(); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda")
accum, G = 2, world // 2
torch.manual_seed(0); lin = nn.Linear(8, 1).to(dev); w0 = [p.detach().clone() for p in lin.parameters()]
X = torch.randn(11, 8, device=dev); y = (torch.rand(11, device=dev) > .5).float(); grp = [list(range(0, 7)), list(range(7, 11))]  # batches of 7 and 4 rows
ref = nn.Linear(8, 1).to(dev); ref.load_state_dict(lin.state_dict())
l = sum(nn.functional.binary_cross_entropy_with_logits(ref(X[b]).squeeze(-1), y[b]) for b in grp) / accum; l.backward()
rn = torch.sqrt(sum((p.grad ** 2).sum() for p in ref.parameters())).item()
opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=0.0)
eng, *_ = deepspeed.initialize(model=lin, optimizer=opt, config={"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "gradient_clipping": 1.0,
    "zero_optimization": {"stage": 2}, "zero_allow_untested_optimizer": True, "steps_per_print": 10**9})
j, part = rank // G, rank % G; rows = np.array_split(np.array(grp[j]), G)[part].tolist()
bce = nn.functional.binary_cross_entropy_with_logits(eng(X[rows]).squeeze(-1), y[rows], reduction="sum")
eng.backward(bce / len(grp[j]) / accum * world)
eng.step(); gn = eng.get_global_grad_norm()
if rank == 0: print(f"GRADCHECK reference grad norm {rn:.6f} | DeepSpeed global grad norm {float(gn):.6f} | ratio {float(gn)/rn:.6f}")
