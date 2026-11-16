# Full fine-tuning (FFT) with DeepSpeed ZeRO-2: negative result

`41_fft_ds.py` is `lora/12_sft.py` with every weight trainable (including the tied token embedding,
389M of 4,022M parameters) on 4 GPUs. Same data (`train_r2.csv`), augmentation seed, batch order, schedule,
clipping and dev-A selection as R2. Each optimizer step still uses 2 token-budget batches, whose rows are
split over 2 ranks each. fp32 master weights, bf16 autocast, head in fp32. lr 1e-5 (head 1e-3), one fixed value.

`ds_gradcheck.py` is a toy check that the loss scaling reproduces the single-process gradient
(DeepSpeed global grad norm / reference = 1.000000).

## Result (one run, as measured)
Training loss stayed at about 0.69 (ln 2) from step 200 to the end; dev-A J at the four evals was 9.19, 3.91,
7.23 and 13.88 (R2: 91.36). The model did not learn. The cause was not tested (candidates: learning rate,
training the embedding with everything else, a script problem). No other configuration was run.

## Run
```bash
# DeepSpeed 0.20.0 has no aarch64 wheel; it is pure Python without compiled ops, so unpack the sdist into ./ds_pkgs
# (plus hjson and msgpack) and put it on PYTHONPATH. Not installed into the main environment.
cd lora && ./setup.sh && cd ..
export MODEL=/path/to/Qwen3Guard-Gen-4B
sbatch fft/run_fft.sbatch                       # 4 GPUs, about 75 minutes
```
Peak memory was about 45 GiB per GPU at 16384 tokens per batch.
