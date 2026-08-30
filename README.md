# Smart MCQ Solver — Qwen2.5-7B-Instruct + LoRA

Fine-tuning `Qwen/Qwen2.5-7B-Instruct` with QLoRA to answer 5-option (A–E)
multiple-choice questions, trained on the Smart MCQ Solver Challenge dataset
and optimized with a custom ranking loss for the competition's MAP@3 metric.

## Overview

- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Fine-tuning method:** QLoRA (4-bit NF4 quantization + LoRA adapters)
- **Task:** Given a question and 5 options (A–E), predict the top-3 most
  likely correct answers, ranked by confidence
- **Target metric:** MAP@3 (Mean Average Precision @ 3)
- **Trainable parameters:** ~40.4M out of ~7.66B total (0.53%)

## Model architecture

The base model is loaded in 4-bit (NF4, double quantization, bfloat16 compute
dtype) via `bitsandbytes`, with LoRA adapters attached to all major linear
projections:

```
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Each option letter (A–E) is verified to tokenize to a single token, and the
model's next-token logits over just those 5 token IDs are used directly as
the option scores — no separate classification head is added.

## Training methodology

### Prompt format

Questions are formatted as a chat-style prompt (Qwen ChatML) asking the model
to respond with a single letter:

```
<|im_start|>system
You are an expert at answering multiple choice questions...
<|im_start|>user
{question}

Options:
A: ...
B: ...
...
<|im_start|>assistant
```

### Custom loss — Weighted RankNet ("FIRST" loss)

Instead of standard cross-entropy over the 5 option logits, training uses a
pairwise ranking loss (RankNet-style) that explicitly pushes the correct
option's score above every incorrect option's score:

```
loss = top1_weight * mean(softplus(-(score_correct - score_wrong)))
```

This aligns the training objective more directly with the MAP@3 ranking
metric than plain classification would. `top1_weight` (default `2.0`)
controls how strongly the correct-vs-wrong margin is emphasized.

### Anti-bias augmentation

The training dataset has an uneven natural distribution of correct answers
(B and C are over-represented). To prevent the model from learning a
positional/letter bias instead of the actual content, `shuffle_options=True`
randomly permutes which letter maps to which option text on every training
example, with `correct_idx` remapped accordingly. This augmentation is
**disabled** for validation/test.

### Hyperparameters

| Setting | Value |
|---|---|
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Learning rate | 2e-4 |
| LR schedule | Cosine with warmup (10%) |
| Max sequence length | 1024 |
| Max grad norm | 1.0 |
| Optimizer | AdamW (β=0.9, 0.999, wd=0.01) |

All hyperparameters live in `config.yaml` and are not hardcoded in the
scripts.

## Dataset

- **Source:** Smart MCQ Solver Challenge (originally a Kaggle competition;
  submissions are now closed)
- **Train set:** 2,000 labeled rows (`id, prompt, A, B, C, D, E, answer`)
- **Test set:** 500 unlabeled rows (`id, prompt, A, B, C, D, E`) — since
  Kaggle submissions are closed, this split can no longer be scored and is
  used only for generating predictions, not evaluation
- Answer distribution is imbalanced (B: 490, C: 459, A: 369, D: 358, E: 324),
  which motivated the option-shuffling augmentation above

Since there is no way to score the official test set anymore, **all
evaluation in this repo is done on a held-out split carved out of
`train.csv`** (see Evaluation below).

## Repository structure

```
my_finetune_project/
├── .gitignore                # excludes data/ and checkpoints/ from the repo
├── README.md
├── config.yaml                # all hyperparameters, paths, wandb settings
├── requirement.txt            # pip dependencies
├── train.py                   # training entrypoint (loads config, trains, saves best checkpoint)
├── test.py                    # inference-only: loads a checkpoint, predicts on test.csv
├── eval.py                    # evaluates a fine-tuned checkpoint on a held-out split of train.csv
├── eval_base_only.py          # evaluates the RAW base model (no LoRA) for baseline comparison
├── data/                       # train.csv / test.csv — gitignored, not in the repo
└── checkpoints/                # saved LoRA adapter weights — gitignored (see Hugging Face section)
```

> `data/` and `checkpoints/` are intentionally excluded from version control
> via `.gitignore` — the dataset is redistributable separately and the
> checkpoint is published on Hugging Face instead (see below) rather than
> committed to git.

## Setup

```bash
conda create -n finetune python=3.10 -y
conda activate finetune
pip install -r requirement.txt
```

Unzip the dataset into `data/` so `data/train.csv` and `data/test.csv` exist,
matching the paths in `config.yaml`.

## Training

```bash
python train.py
```

- If `full_train: true` in `config.yaml`, the model trains on all rows of
  `train.csv` (no held-out validation — used for producing a final
  deployment checkpoint, not for measuring generalization).
- If `full_train: false`, a real `val_frac` split (default 10%) is held out
  and never trained on; `Val MAP@3` is printed after every epoch and is a
  trustworthy generalization estimate.
- The best checkpoint (by validation MAP@3) is saved to `save_path`.
- Training logs (batch loss, LR, epoch metrics) are sent to Weights & Biases
  if `use_wandb: true`.

## Evaluation

Because the official test set can no longer be scored, evaluation is done
against a held-out sample of `train.csv`'s labeled rows.

```bash
# Evaluate the fine-tuned checkpoint
python eval.py --val_frac 0.15

# Evaluate the raw base model (no LoRA) for comparison
python eval_base_only.py --val_frac 0.15
```

**Important caveat:** if a checkpoint was trained with `full_train: true`,
it has already seen every row in `train.csv`, including whatever rows get
sampled into the "held-out" eval split — so `eval.py` on such a checkpoint
measures memorization, not generalization. For an honest score, evaluate a
checkpoint that was trained with `full_train: false` on a split that
excludes the rows it was evaluated on.

### Results

**Kaggle private leaderboard (MAP@3) — the authoritative, held-out score:**

| Model | MAP@3 (private LB) |
|---|---|
| Base model (no LoRA) | 0.65 |
| Fine-tuned (LoRA) | **0.78** |

Fine-tuning improved MAP@3 by **+0.13** on data the model never saw during
training or development — this is the trustworthy, final result.

**Local held-out evaluation (sample from `train.csv`, for quick iteration):**

| Model | MAP@3 | Accuracy | F1 (macro) | Notes |
|---|---|---|---|---|
| Base model (no LoRA) | 0.8517 | 0.7433 | 0.7371 | Zero-shot, 300-row sample |
| Fine-tuned (full_train=True) | 1.0000 | 1.0000 | 1.0000 | Not meaningful — model memorized this data during training; kept here only to illustrate why `eval.py`/`eval_base_only.py` need a genuinely held-out split to be trusted |

The local eval scripts (`eval.py`, `eval_base_only.py`) are useful for fast
iteration without needing a Kaggle submission, but the Kaggle private
leaderboard numbers above are the real measure of generalization, since that
test set was never touched during training or local validation.

## Uploading to Hugging Face

Since only the LoRA adapter needs to be shared (not the full base model),
push just the adapter directory:

```bash
pip install huggingface_hub
huggingface-cli login
```

Then, either via the CLI:

```bash
huggingface-cli upload <your-username>/<repo-name> ./checkpoints/first_mcq_lora
```

or from Python, using PEFT's built-in helper (loads the adapter, then pushes it):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "./checkpoints/first_mcq_lora")
model.push_to_hub("<your-username>/<repo-name>")
```

Include a `base_model: Qwen/Qwen2.5-7B-Instruct` field and the `library_name: peft`
tag in the Hugging Face model card's YAML frontmatter so the Hub correctly
links it as an adapter for the base model.

## Notes on reproducibility

- Seed is fixed (`seed: 42` in `config.yaml`) for the train/val split and
  all RNG (Python, NumPy, PyTorch, CUDA).
- `full_train` controls whether validation is real or a small dummy sample
  — check this setting before trusting any printed metric.
