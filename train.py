"""
Training script for Qwen2.5-7B-Instruct LoRA fine-tuning on the
Smart MCQ Solver Challenge dataset.

Multi-GPU (DDP) version via HuggingFace Accelerate.

Usage:
    accelerate config                 # one-time, or use the CLI flags below
    accelerate launch --multi_gpu --num_processes=7 train.py
"""
import os
import gc
import random
import yaml
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
from sklearn.metrics import f1_score
from accelerate import Accelerator
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OPTION_KEYS = ["A", "B", "C", "D", "E"]


# ─────────────────────────────────────────────────────────
# 0. CONFIG LOADING (from config.yaml)
# ─────────────────────────────────────────────────────────

with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

CFG = SimpleNamespace(
    model_name      = _cfg["model"]["name"],
    save_path       = _cfg["model"]["save_path"],

    train_csv       = _cfg["data"]["train_csv"],
    test_csv        = _cfg["data"]["test_csv"],
    submission_csv  = _cfg["data"]["submission_csv"],
    val_frac        = _cfg["data"]["val_frac"],
    max_length      = _cfg["data"]["max_length"],

    lora_r          = _cfg["lora"]["r"],
    lora_alpha      = _cfg["lora"]["alpha"],
    lora_dropout    = _cfg["lora"]["dropout"],

    num_epochs      = _cfg["training"]["num_epochs"],
    batch_size      = _cfg["training"]["batch_size"],   # NOTE: this is now PER-GPU batch size
    grad_accum      = _cfg["training"]["grad_accum"],
    lr              = float(_cfg["training"]["lr"]),
    warmup_ratio    = _cfg["training"]["warmup_ratio"],
    max_grad_norm   = _cfg["training"]["max_grad_norm"],
    top1_weight     = _cfg["training"]["top1_weight"],
    full_train      = _cfg["training"]["full_train"],
    shuffle_options = _cfg["training"]["shuffle_options"],

    use_wandb       = _cfg["wandb"]["use_wandb"],
    wandb_project   = _cfg["wandb"]["project"],
    wandb_run_name  = _cfg["wandb"]["run_name"],

    seed            = _cfg["seed"],
)


# ─────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────
# 2. PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at answering multiple choice questions. "
    "Given a question and five options, identify the most likely correct answer. "
    "Respond with only a single letter: A, B, C, D, or E."
)


def build_prompt(prompt_text: str, options: dict) -> str:
    """
    prompt_text : value from the 'prompt' column (already has question text)
    options     : dict {A: text, B: text, ...} — may be permuted during training
    """
    option_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt_text}\n\n"
        f"Options:\n{option_str}\n\n"
        f"Which option is correct? Answer with a single letter.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────

class MCQDataset(Dataset):
    """
    has_labels=True  : expects 'answer' column (train / val)
    has_labels=False : no answer column needed (test / inference)

    shuffle_options=True (training only):
        Randomly remaps which letter gets which option text.
        correct_idx updated accordingly.
        Prevents the model from learning letter-position bias.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        has_labels: bool = True,
        shuffle_options: bool = False,
    ):
        self.df              = df.reset_index(drop=True)
        self.tokenizer       = tokenizer
        self.max_length      = max_length
        self.has_labels      = has_labels
        self.shuffle_options = shuffle_options

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        option_texts = [str(row[k]) for k in OPTION_KEYS]

        if self.has_labels:
            correct_pos = OPTION_KEYS.index(str(row["answer"]).strip().upper())
        else:
            correct_pos = 0  # placeholder, never used

        if self.shuffle_options and self.has_labels:
            perm        = list(range(5))
            random.shuffle(perm)
            shuffled    = [option_texts[p] for p in perm]
            options     = dict(zip(OPTION_KEYS, shuffled))
            correct_idx = perm.index(correct_pos)
        else:
            options     = dict(zip(OPTION_KEYS, option_texts))
            correct_idx = correct_pos

        prompt   = build_prompt(str(row["prompt"]), options)
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.has_labels:
            item["correct_idx"] = torch.tensor(correct_idx, dtype=torch.long)

        return item


def collate_fn(batch, pad_token_id: int):
    """
    Left-pad all sequences to the longest in the batch.
    Left-padding means logits[:, -1, :] always points to the last
    REAL token position.
    """
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids_list = []
    attn_mask_list = []

    for x in batch:
        pad_len = max_len - x["input_ids"].size(0)
        input_ids_list.append(torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
            x["input_ids"],
        ]))
        attn_mask_list.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            x["attention_mask"],
        ]))

    out = {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
    }
    if "correct_idx" in batch[0]:
        out["correct_idx"] = torch.stack([x["correct_idx"] for x in batch])

    return out


# ─────────────────────────────────────────────────────────
# 4. MODEL SETUP
# ─────────────────────────────────────────────────────────

def setup_model_and_tokenizer(model_name: str, accelerator: Accelerator):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # IMPORTANT: with DDP + bitsandbytes, do NOT use device_map="auto".
    # "auto" will try to shard the model across every visible GPU inside
    # EACH process, which conflicts with Accelerate's own process-per-GPU
    # DDP setup. Pin each process to exactly one GPU instead.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Gradient checkpointing trades compute for activation memory — needed
    # here because 7B backward-pass activations don't fit in 10.75 GiB
    # even at batch_size=1. use_reentrant=False plays nicer with PEFT.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # required so PEFT's frozen base layers still get gradient flow through checkpointed activations
    model.config.use_cache = False      # incompatible with grad checkpointing; also unnecessary during training

    if accelerator.is_main_process:
        model.print_trainable_parameters()
    return model, tokenizer


def get_option_token_ids(tokenizer) -> torch.Tensor:
    """
    Returns tensor of shape (5,) with token IDs for A, B, C, D, E.
    Asserts each letter is a single token — will loudly fail if not.
    """
    ids = []
    for letter in OPTION_KEYS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"'{letter}' tokenises to {len(token_ids)} tokens: {token_ids}.\n"
            f"Fix: try encoding ' {letter}' (space prefix) and update get_option_token_ids."
        )
        ids.append(token_ids[0])
    return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────
# 5. FIRST LOSS — Weighted RankNet, NaN-safe
# ─────────────────────────────────────────────────────────

def weighted_ranknet_loss(
    option_logits: torch.Tensor,   # (B, 5) — may be bfloat16 from model
    correct_idx:   torch.Tensor,   # (B,)
    top1_weight:   float = 2.0,
) -> torch.Tensor:

    # float32 — MUST happen before any arithmetic
    option_logits = option_logits.float()

    # kill any NaN / inf from 4-bit model
    option_logits = torch.nan_to_num(
        option_logits, nan=0.0, posinf=100.0, neginf=-100.0
    )

    # clamp (softplus is fine up to ~85 in float32, but be safe)
    option_logits = torch.clamp(option_logits, min=-100.0, max=100.0)

    B, N = option_logits.shape

    # One-hot mask for correct option  (B, 5)
    correct_mask = F.one_hot(correct_idx, num_classes=N).float()

    # Correct option score for each sample  (B, 1)
    correct_scores = (option_logits * correct_mask).sum(dim=1, keepdim=True)

    # Margin: s_correct - s_j for all j  (B, 5)
    margins = correct_scores - option_logits

    # RankNet pair loss = softplus(-margin) = log(1 + exp(-margin))
    pair_losses = F.softplus(-margins)

    # Zero out the self-pair (correct vs itself)
    pair_losses = pair_losses * (1.0 - correct_mask)

    if torch.isnan(pair_losses).any():
        pair_losses = torch.nan_to_num(pair_losses, nan=0.0)

    loss = top1_weight * pair_losses.sum() / B
    return loss


# ─────────────────────────────────────────────────────────
# 6. MAP@3 METRIC
# ─────────────────────────────────────────────────────────

def compute_map3(preds: list, labels: list) -> float:
    scores = []
    for pred, label in zip(preds, labels):
        ap = 0.0
        for k, p in enumerate(pred[:3]):
            if p == label:
                ap = 1.0 / (k + 1)
                break
        scores.append(ap)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────
# 7. EVALUATION LOOP
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, option_token_ids: torch.Tensor, accelerator: Accelerator) -> float:
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in tqdm(loader, desc="  Evaluating", leave=False, disable=not accelerator.is_main_process):
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        correct_idx    = batch["correct_idx"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        last_logits   = outputs.logits[:, -1, :].float()
        last_logits   = torch.nan_to_num(last_logits, nan=0.0)
        option_logits = last_logits[:, option_token_ids]

        # Gather predictions + labels across all processes before scoring
        top3       = torch.argsort(option_logits, dim=-1, descending=True)[:, :3]
        top3_g     = accelerator.gather_for_metrics(top3)
        labels_g   = accelerator.gather_for_metrics(correct_idx)

        all_preds.extend(top3_g.cpu().numpy().tolist())
        all_labels.extend(labels_g.cpu().numpy().tolist())

    return compute_map3(all_preds, all_labels)


# ─────────────────────────────────────────────────────────
# 8. TRAINING
# ─────────────────────────────────────────────────────────

def train(model, tokenizer, train_df: pd.DataFrame, val_df: pd.DataFrame, accelerator: Accelerator):
    pad_id           = tokenizer.pad_token_id
    option_token_ids = get_option_token_ids(tokenizer).to(accelerator.device)

    train_ds = MCQDataset(
        train_df, tokenizer,
        max_length=CFG.max_length,
        has_labels=True,
        shuffle_options=CFG.shuffle_options,
    )
    val_ds = MCQDataset(
        val_df, tokenizer,
        max_length=CFG.max_length,
        has_labels=True,
        shuffle_options=False,
    )

    # NOTE: batch_size here is now PER-GPU. Effective global batch size =
    # batch_size * num_processes * grad_accum.
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    total_steps  = (len(train_loader) // CFG.grad_accum) * CFG.num_epochs
    warmup_steps = int(total_steps * CFG.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Hand everything to Accelerate. It wraps `model` in DDP, shards the
    # dataloaders across processes, and moves tensors to the right device.
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_map3   = 0.0
    global_step = 0

    if accelerator.is_main_process:
        print(f"\nTraining | epochs={CFG.num_epochs} | "
              f"steps/epoch={len(train_loader)} | "
              f"per_gpu_batch={CFG.batch_size} | num_gpus={accelerator.num_processes} | "
              f"effective_batch={CFG.batch_size * CFG.grad_accum * accelerator.num_processes}\n")

    for epoch in range(CFG.num_epochs):
        model.train()
        running_loss = 0.0
        nan_batches  = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.num_epochs}",
                    disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            correct_idx    = batch["correct_idx"]

            # Accelerate's grad-accum context handles no_sync() for you,
            # avoiding an all-reduce on every micro-step.
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                last_logits   = outputs.logits[:, -1, :]
                option_logits = last_logits[:, option_token_ids]

                loss = weighted_ranknet_loss(
                    option_logits,
                    correct_idx,
                    top1_weight=CFG.top1_weight,
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    if accelerator.is_main_process:
                        print(f"\n  [WARNING] NaN/Inf loss at step {step} — skipping batch")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            avg_loss      = running_loss / (step + 1)
            global_step  += 1

            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{avg_loss:.4f}", nan_skip=nan_batches)

                if CFG.use_wandb:
                    wandb.log({
                        "batch/loss": loss.item(),
                        "batch/avg_loss": avg_loss,
                        "batch/lr":   scheduler.get_last_lr()[0],
                        "batch/nan_skipped": nan_batches,
                        "global_step": global_step,
                    })

        avg_loss = running_loss / max(len(train_loader) - nan_batches, 1)

        # Validate
        map3 = evaluate(model, val_loader, option_token_ids, accelerator)

        # Val accuracy + F1
        model.eval()
        all_preds_flat  = []
        all_labels_flat = []

        with torch.no_grad():
            for vbatch in val_loader:
                vi      = vbatch["input_ids"]
                vm      = vbatch["attention_mask"]
                vc      = vbatch["correct_idx"]
                vout    = model(input_ids=vi, attention_mask=vm)
                vlogits = vout.logits[:, -1, :].float()
                vlogits = torch.nan_to_num(vlogits, nan=0.0)
                vopt    = vlogits[:, option_token_ids]
                vpred   = vopt.argmax(dim=-1)

                vpred_g = accelerator.gather_for_metrics(vpred)
                vc_g    = accelerator.gather_for_metrics(vc)
                all_preds_flat.extend(vpred_g.cpu().numpy().tolist())
                all_labels_flat.extend(vc_g.cpu().numpy().tolist())

        val_acc = np.mean(np.array(all_preds_flat) == np.array(all_labels_flat))
        val_f1  = f1_score(all_labels_flat, all_preds_flat, average="macro")

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val MAP@3: {map3:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"NaN batches skipped: {nan_batches}")

            if CFG.use_wandb:
                wandb.log({
                    "epoch/train_loss": avg_loss,
                    "epoch/val_map3":   map3,
                    "epoch/val_acc":    val_acc,
                    "epoch/val_f1":     val_f1,
                    "epoch":            epoch + 1,
                    "global_step":      global_step,
                })

        accelerator.wait_for_everyone()
        if map3 > best_map3:
            best_map3 = map3
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(CFG.save_path)
                tokenizer.save_pretrained(CFG.save_path)
                print(f"  ✓ Saved best checkpoint  (MAP@3 = {best_map3:.4f})\n")

    if accelerator.is_main_process:
        print(f"\nDone. Best Val MAP@3 = {best_map3:.4f}")
    return model


# ─────────────────────────────────────────────────────────
# 9. SANITY CHECK
# ─────────────────────────────────────────────────────────

def sanity_check(tokenizer):
    print("\n" + "=" * 55)
    print("SANITY CHECK")
    print("=" * 55)

    print("\n[1] Option letter → single token?")
    all_ok = True
    for letter in OPTION_KEYS:
        ids    = tokenizer.encode(letter, add_special_tokens=False)
        status = "✓ OK" if len(ids) == 1 else f"✗ FAIL ({len(ids)} tokens)"
        print(f"    '{letter}' → token_ids={ids}  [{status}]")
        if len(ids) != 1:
            all_ok = False
    if not all_ok:
        raise RuntimeError(
            "Some option letters tokenise to multiple tokens. "
            "Try ' A', ' B', ... (space prefix) and update get_option_token_ids."
        )

    print("\n[2] Sample prompt preview:")
    sample_opts   = {k: f"This is option {k} text." for k in OPTION_KEYS}
    sample_prompt = build_prompt("What is 2 + 2?", sample_opts)
    print(sample_prompt)

    print("\n[3] Loss on dummy data (must be finite, non-NaN):")
    logits = torch.randn(4, 5)
    cidx   = torch.tensor([0, 1, 2, 3])
    loss   = weighted_ranknet_loss(logits, cidx)
    print(f"    loss={loss.item():.4f}  nan={torch.isnan(loss).item()}  "
          f"inf={torch.isinf(loss).item()}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss sanity check failed!"

    print("\nAll checks passed ✓\n" + "=" * 55 + "\n")


# ─────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=1)  # we handle accum via CFG.grad_accum inside accumulate()

    set_seed(CFG.seed)

    train_df = pd.read_csv(CFG.train_csv)
    test_df  = pd.read_csv(CFG.test_csv)
    if accelerator.is_main_process:
        print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
        print(f"Answer distribution:\n{train_df['answer'].value_counts()}\n")

    assert all(c in train_df.columns for c in ["id", "prompt", "A", "B", "C", "D", "E", "answer"]), \
        f"Train columns mismatch: {list(train_df.columns)}"
    assert all(c in test_df.columns for c in ["id", "prompt", "A", "B", "C", "D", "E"]), \
        f"Test columns mismatch: {list(test_df.columns)}"

    if CFG.full_train:
        if accelerator.is_main_process:
            print("full_train=True — training on all rows")
        val_df   = train_df.sample(n=20, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        if accelerator.is_main_process:
            print(f"Train: {len(train_df)} | Val (dummy): {len(val_df)}")
    else:
        val_df   = train_df.sample(frac=CFG.val_frac, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        if accelerator.is_main_process:
            print(f"Split → Train: {len(train_df)} | Val: {len(val_df)}")

    if CFG.use_wandb and accelerator.is_main_process:
        # Run `wandb login` once in your terminal before this (see setup notes).
        wandb.init(
            project=CFG.wandb_project,
            name=CFG.wandb_run_name,
            config={
                "model":       CFG.model_name,
                "epochs":      CFG.num_epochs,
                "batch_size":  CFG.batch_size,
                "grad_accum":  CFG.grad_accum,
                "lr":          CFG.lr,
                "lora_r":      CFG.lora_r,
                "lora_alpha":  CFG.lora_alpha,
                "max_length":  CFG.max_length,
                "top1_weight": CFG.top1_weight,
                "full_train":  CFG.full_train,
                "num_gpus":    accelerator.num_processes,
            }
        )

    model, tokenizer = setup_model_and_tokenizer(CFG.model_name, accelerator)
    if accelerator.is_main_process:
        sanity_check(tokenizer)
    accelerator.wait_for_everyone()

    model = train(model, tokenizer, train_df, val_df, accelerator)

    if CFG.use_wandb and accelerator.is_main_process:
        wandb.finish()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    accelerator.wait_for_everyone()

    # Run inference on a single process only — no need to duplicate this 7x.
    if accelerator.is_main_process:
        print("\nLoading best checkpoint for inference...")
        from test import load_model, predict as run_predict

        best_model, tokenizer = load_model(CFG.model_name, CFG.save_path)
        submission = run_predict(best_model, tokenizer, test_df)
        submission.to_csv(CFG.submission_csv, index=False)

        print(f"\nSubmission saved → {CFG.submission_csv}")
        print(submission.head(10).to_string(index=False))
