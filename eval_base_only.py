"""
Baseline evaluation script — measures MAP@3 / Accuracy / F1 of the
RAW base model (Qwen2.5-7B-Instruct), with NO LoRA adapter applied.

Use this to compare against your fine-tuned checkpoint's score from
eval.py, so you can see how much the fine-tuning actually helped.

This script is fully standalone — it does not import or modify
test.py or eval.py.

Usage:
    python eval_base_only.py                # uses val_frac from config.yaml
    python eval_base_only.py --val_frac 0.15
"""

import os
import argparse
import yaml
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from train import MCQDataset, collate_fn, compute_map3, OPTION_KEYS, set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_frac", type=float, default=None,
                    help="Fraction of train.csv to hold out for eval (default: from config.yaml)")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for the split (default: from config.yaml)")
    return p.parse_args()


def load_base_model_only(model_name):
    """Loads ONLY the base model — no LoRA adapter attached."""
    print(f"Loading base model (no LoRA): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Base model ready.")
    return model, tokenizer


def get_option_token_ids(tokenizer):
    ids = []
    for letter in OPTION_KEYS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(token_ids) == 1, f"'{letter}' → {token_ids} (not single token)"
        ids.append(token_ids[0])
    print(f"Option token IDs: { {k: v for k, v in zip(OPTION_KEYS, ids)} }")
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def run_eval(model, tokenizer, val_df, max_length, batch_size, device):
    option_token_ids = get_option_token_ids(tokenizer).to(device)
    pad_id           = tokenizer.pad_token_id

    val_ds = MCQDataset(
        val_df, tokenizer,
        max_length=max_length,
        has_labels=True,
        shuffle_options=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
    )

    all_top3   = []
    all_top1   = []
    all_labels = []

    for batch in tqdm(val_loader, desc="Evaluating (base model, no LoRA)"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        correct_idx    = batch["correct_idx"]

        outputs       = model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits   = outputs.logits[:, -1, :].float()
        last_logits   = torch.nan_to_num(last_logits, nan=0.0)
        option_logits = last_logits[:, option_token_ids]

        top3 = torch.argsort(option_logits, dim=-1, descending=True)[:, :3]
        top1 = option_logits.argmax(dim=-1)

        all_top3.extend(top3.cpu().numpy().tolist())
        all_top1.extend(top1.cpu().numpy().tolist())
        all_labels.extend(correct_idx.numpy().tolist())

    map3     = compute_map3(all_top3, all_labels)
    accuracy = accuracy_score(all_labels, all_top1)
    f1       = f1_score(all_labels, all_top1, average="macro")

    return {
        "map3": map3,
        "accuracy": accuracy,
        "f1_macro": f1,
        "n_examples": len(all_labels),
    }


if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        _cfg = yaml.safe_load(f)

    CFG = SimpleNamespace(
        model_name = _cfg["model"]["name"],
        train_csv  = _cfg["data"]["train_csv"],
        max_length = _cfg["data"]["max_length"],
        val_frac   = args.val_frac if args.val_frac is not None else _cfg["data"]["val_frac"],
        seed       = args.seed if args.seed is not None else _cfg["seed"],
    )

    set_seed(CFG.seed)

    train_df = pd.read_csv(CFG.train_csv)
    val_df   = train_df.sample(frac=CFG.val_frac, random_state=CFG.seed).reset_index(drop=True)
    print(f"Loaded train.csv: {len(train_df)} rows | Held-out eval split: {len(val_df)} rows")

    model, tokenizer = load_base_model_only(CFG.model_name)
    device = next(model.parameters()).device

    results = run_eval(model, tokenizer, val_df, CFG.max_length, batch_size=4, device=device)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS — BASE MODEL (no LoRA)")
    print("=" * 40)
    print(f"  Examples : {results['n_examples']}")
    print(f"  MAP@3    : {results['map3']:.4f}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print("=" * 40)
