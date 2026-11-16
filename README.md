# HalmstadLangCode: Halmstad team @ LangCode 2027, Shared Task 2

Binary safety classification (S safe / U unsafe) of prompts in 10 languages, scored by
J = R_U + R_S - 100. Model: Qwen3Guard-Gen-4B with a new linear head on the last templated token.

| folder | content |
|---|---|
| `lora/` | LoRA classifier fine-tuning (the R2 model) and everything needed to recreate it |
| `fft/` | full fine-tuning with DeepSpeed ZeRO-2 (negative result) |
| `data/` | task data (`train.csv`, `dev.csv`, `devtest.csv`), the enlarged training sets `train_r1.csv`/`train_r2.csv` built from `train.csv` only, the twin-aware dev split and the minimal-pair id list. `SHA256SUMS` lists the files used. |

## Results (single seed 0; dev-A selects checkpoint and threshold, dev-B is only scored)
| model | dev-B J | note |
|---|---|---|
| msaug (LoRA, `train.csv` + many-shot aug) | 91.80 | reference |
| r1 (LoRA, `train_r1.csv`) | 92.04 | not significantly above msaug (paired bootstrap CI spans 0) |
| **r2 (LoRA, `train_r2.csv`)** | **92.52** | r2 - msaug +0.71 [-0.20, +1.62]; r2 - r1 +0.47 [-0.42, +1.34] |
| fft (full fine-tune, lr 1e-5) | not usable | loss stayed at ln 2, dev-A J 13.9 |

The r2 devtest score reported by the task's scorer was J 92.96. The dev-B differences between
msaug, r1 and r2 are within noise. Robustness transformations are assumed label-preserving, not verified.

## Recreate R2 (`lora/`)
Training uses `train_r2.csv`, which is built from `train.csv` only. Never train on dev or devtest.

```bash
pip install -r lora/requirements.txt          # aarch64 GH200 build used; adapt torch to your platform
# download Qwen3Guard-Gen-4B into a local folder and: export MODEL=/path/to/Qwen3Guard-Gen-4B
cd lora && ./setup.sh                          # links ../data, checks SHA-256
# optional: rebuild what is already in data/ (outputs should match SHA256SUMS)
python 30_group_split.py                       # writes dev_split_group.json (twin-aware dev-A/dev-B split, seed 0)
python 31_build_train.py                       # writes train_r1.csv, train_r2.csv
python 10_eval_sets.py                         # robustness sets in evalsets/ (scored on dev-B only)
# train R2 (1 GPU, about 4 h on a GH200); see run_r2.sbatch
python 12_sft.py --model "$MODEL" --tag r2 --train_csv train_r2.csv --many_shot_aug 0.3
# smoke test first: add  --tag smoke --limit_train 200 --limit_eval 200 --epochs 1
```
`12_sft.py` writes `sft_r2/best/` (adapter, head, dev-A threshold) and `scores_sft_r2.npz`
(dev, robustness sets and devtest scores). `40_make_submissions.py` builds the submission zips from
the saved scores (threshold = best J on dev-A, devtest only predicted); it expects `scores_sft_r1.npz`
and `scores_sft_r2.npz`, so train `--tag r1 --train_csv train_r1.csv --many_shot_aug 0.3` too, or edit the model list.

Fixed settings: LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down; lr 1e-4 (head 1e-3); 2 epochs;
2 dev-A evals per epoch; `--max_tokens 32768`; 16384 tokens per batch, 2 batches per step; many-shot
augmentation 0.3. No hyperparameter sweeps.

The model id of the local 4B copy was not recorded with its hub id; `Qwen/Qwen3Guard-Gen-4B` is expected.

## Full fine-tuning (`fft/`)
See `fft/README.md`.

## Data note
`data/` holds the shared-task data, which includes harmful-request and attack-style prompts used to train
and evaluate the classifier. Keep the repository private unless the task's data terms allow redistribution.
