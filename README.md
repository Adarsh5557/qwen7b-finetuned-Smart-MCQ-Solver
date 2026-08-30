# Smart MCQ Solver

Fine-tuning `Qwen2.5-7B-Instruct` to answer 5-option (A-E) multiple-choice
questions, trained with QLoRA and a custom ranking loss aimed directly at
the competition's actual metric (MAP@3), instead of plain next-token
cross-entropy.

**[Try it live](https://huggingface.co/spaces/Adarshraj31415/smart-mcq-solver-demo)**
**[LoRA adapter on the Hub](https://huggingface.co/Adarshraj31415/smart-mcq-qwen2.5-lora)**

**TL;DR:** base model scores 0.65 MAP@3 on the Kaggle private leaderboard.
Fine-tuned model scores **0.78**. That's a 0.13 improvement from about 40M
trainable parameters (0.53% of the model) and one epoch over 1,800 examples.

---

## Why this exists

The task: given a question and 5 labeled options, return your top-3 guesses
ranked by confidence, scored by MAP@3 (full credit if the right answer is
your #1 guess, half credit if it's #2, a third credit if it's #3, zero
otherwise). This is a Kaggle competition format (Smart MCQ Solver
Challenge). Submissions are now closed, so the private leaderboard score
quoted above is the last real signal I got from Kaggle before it locked.

Everything past that point, local evaluation scripts, the retraining runs,
the multi-GPU debugging, happened after the leaderboard was already frozen.
That's worth keeping in mind when you read the results section: the
0.65 to 0.78 number is real and trustworthy. Numbers from my own local eval
script need an asterisk, explained below.

## How it works

**Base model:** `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit NF4
(double-quantized, bfloat16 compute dtype) via `bitsandbytes`.

**Adapter:** LoRA, r=16, alpha=32, dropout=0.05, attached to every linear
projection in the transformer block (`q/k/v/o_proj` and
`gate/up/down_proj`), not just attention. About 40.4M trainable params out
of about 7.66B total.

**Scoring trick:** rather than generating text and parsing it, I read the
model's next-token logits at the single position right after the prompt,
restricted to just the 5 token IDs for "A" through "E" (each verified to be
a single token, see the sanity check in `train.py`). Softmax over those 5
logits gives a probability distribution over options directly, no
generation or parsing needed. This is both faster and removes an entire
class of "model said the right thing but I parsed it wrong" bugs.

**Loss: weighted RankNet, not cross-entropy.**

```
loss = top1_weight * mean( softplus(-(score_correct - score_wrong)) )
       over all wrong options, per example
```

Cross-entropy optimizes for the correct answer having the single highest
probability. MAP@3 only cares that the correct answer is somewhere in the
top 3. A pairwise ranking loss, pushing the correct option's score above
every wrong option's score individually, is a closer match to what's
actually being scored. `top1_weight=2.0` upweights getting rank 1 right
specifically, since that's worth more points than rank 2 or 3.

**Anti-bias augmentation:** the raw label distribution is skewed (B: 490,
C: 459, A: 369, D: 358, E: 324 out of 2000). Left alone, a model can partially
game this by learning "B and C are more likely" instead of reading the
question. The fix: during training only, randomly permute which letter maps
to which option text on every example, and remap the correct index to
match. The model never gets to memorize a positional prior, it has to read.
This is disabled at eval and inference time.

## The multi-GPU saga (why "just add more GPUs" isn't free)

This section exists because I hit the same wall twice and want the next
person (possibly future me) to skip both trips.

**Attempt 1:** `python train.py` on a single GPU, a 2080 Ti with 10.75 GiB
VRAM. Trained fine for 10 steps, then:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 6 has a total capacity of 10.75 GiB of which 23.62 MiB is free.
```

The tell here isn't that it failed immediately (that would mean the model
just doesn't fit), it's that it failed at step 10. That pattern points at
allocator fragmentation, not a hard capacity wall: activation memory for a
7B model's backward pass, combined with bitsandbytes dequantizing 4-bit
weights on the fly per layer, adds up fast with no gradient checkpointing
enabled.

**Attempt 2:** switched to multi-GPU with `accelerate`:

```bash
accelerate launch --multi_gpu --num_processes=7 --mixed_precision=bf16 train.py
```

This part is worth being explicit about, because it's a common
misconception: multi-GPU does not reduce each GPU's own memory load. Each
of the 7 processes still runs its own independent `batch_size=1`
forward/backward pass on its own GPU. You get 7x throughput (7 examples
processed in parallel instead of 1), not more headroom per GPU. So this run
OOM'd too, same failure, now visible on rank 4 in the DDP logs instead of a
single process.

**The actual fix: gradient checkpointing.**

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # required for PEFT + grad checkpointing together
```

This trades compute for memory: instead of storing every layer's
activations for the backward pass, it recomputes them on the fly. Slower
per step, dramatically lower peak memory. Combined with multi-GPU for
throughput, this is what finally got a clean run: 258 steps in 15 minutes
23 seconds across 7 GPUs, no OOM.

Lesson, stated plainly: multi-GPU solves utilization, gradient checkpointing
solves memory. They are not substitutes for each other. If you only have
room in your head for one fix, figure out which problem you actually have
before reaching for either.

## Known limitation: don't trust my local eval numbers yet

After the multi-GPU run above finished, `eval.py --val_frac 0.15` reported
99.5% MAP@3 and 99.33% accuracy on a 300-example sample. That number should
raise your eyebrows. It raised mine. 99%+ on held-out MCQ data after a
single epoch is not a realistic generalization result.

The likely cause: `train.py` holds out its validation split with
`train_df.sample(frac=0.1, random_state=42)` (200 of 2000 rows), training on
the remaining 1800. But `eval.py`, when run separately, does an independent
`train_df.sample(frac=0.15, random_state=42)` against the same full
2000-row `train.csv`, not a file guaranteed disjoint from what training
actually saw. With only 200 rows ever excluded from training, a fresh
random 300-row sample will, by simple probability, contain mostly rows the
model already trained on. That's not measuring generalization, it's mostly
re-scoring the training set.

This is a real bug in `eval.py`, not yet fixed as of this README. The
Kaggle private leaderboard score (0.78, quoted above) is unaffected by
this, that test set was genuinely never touched by training, at any point.
But any number from `eval.py` or `eval_base_only.py` should be treated as
provisional until the split logic is fixed to persist and reuse the exact
row indices `train.py` excluded, rather than re-sampling independently.

If you're picking this project up: the fix is to save the held-out index
list (for example `val_indices.json`) at the end of `train.py`'s split
step, and have `eval.py` load and reuse those exact indices instead of
calling `.sample()` again with just a matching seed. A matching seed on a
different sample size (0.1 vs 0.15) does not guarantee the same rows get
excluded. That's the actual bug.

## Results

| Model | MAP@3 | Source | Trustworthy |
|---|---|---|---|
| Base model (no LoRA) | 0.65 | Kaggle private leaderboard | Yes |
| Fine-tuned (LoRA) | 0.78 | Kaggle private leaderboard | Yes |
| Base model (no LoRA) | 0.8517 | Local eval, 300-row sample | Yes (base model never trained on anything) |
| Fine-tuned (LoRA) | 0.995 | Local eval, 300-row sample | No, see limitation above, likely leakage |

The number that matters is the 0.13 improvement on Kaggle's private test
set. Everything else here is a debugging trail, kept visible on purpose.

## Repository structure

```
my_finetune_project/
├── .gitignore              # excludes data/ and checkpoints/
├── README.md
├── config.yaml              # all hyperparameters, paths, wandb settings
├── requirement.txt          # pip dependencies
├── train.py                 # training entrypoint, supports single and multi-GPU via accelerate
├── test.py                  # inference-only: loads a checkpoint, predicts on test.csv
├── eval.py                  # evaluates a fine-tuned checkpoint on a sampled split of train.csv (see limitation above)
├── eval_base_only.py        # evaluates the raw base model (no LoRA) for baseline comparison
├── data/                     # train.csv / test.csv, gitignored
└── checkpoints/              # saved LoRA adapter weights, gitignored, published to HF Hub instead
```

## Running it yourself

```bash
conda create -n finetune python=3.10 -y
conda activate finetune
pip install -r requirement.txt
```

Single GPU:
```bash
python train.py
```

Multi-GPU (recommended if you have more than one card, see the saga above
for why gradient checkpointing matters either way):
```bash
accelerate launch --multi_gpu --num_processes=<N> --mixed_precision=bf16 train.py
```

## Live demo

The [Space](https://huggingface.co/spaces/Adarshraj31415/smart-mcq-solver-demo)
runs on Hugging Face ZeroGPU, free shared GPU access, attached only for the
duration of each request via the `@spaces.GPU` decorator. The first request
after the Space goes idle will be slower (model loads fresh and gets
cached); subsequent requests reuse the loaded model.

To use the adapter directly in your own code instead of the demo:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "Adarshraj31415/smart-mcq-qwen2.5-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

## What I'd do differently next time

- Persist validation row indices to disk instead of re-deriving them by
  seed and fraction in a second script. The eval leakage bug above would
  have been impossible if the split were saved once and loaded everywhere.
- Enable gradient checkpointing from the start rather than discovering the
  need for it via two separate OOM crashes.
- Get a second, truly held-out labeled split before the Kaggle leaderboard
  closes, specifically so local evaluation doesn't depend on re-sampling
  the only labeled file that exists.
