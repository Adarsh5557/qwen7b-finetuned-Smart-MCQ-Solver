"""
Evaluation-only script — measures MAP@3 / Accuracy / F1 of the saved
LoRA checkpoint on a held-out split of train.csv.

Why train.csv and not test.csv?
    test.csv (from the Kaggle competition) has no 'answer' column —
    it was only ever meant to be scored by Kaggle's submission system.
    Since submissions are closed, the only labeled data you have is
    train.csv, so we carve out a fresh validation split from it here.

    NOTE: if you trained with full_train=True, the model has already
    seen ALL of train.csv during training. Metrics computed on it will
    be optimistic (the model has memorized some of these examples).
    For a more honest number, retrain with full_train=False and a
    real val_frac, then re-run this script.

Usage:
    python eval.py                  # uses val_frac from config.yaml
    python eval.py --val_frac 0.2   # override the split size
    python eval.py --seed 123       # different random split
"""

import os
import argparse
import yaml
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from test import load_model, get_option_token_ids
from train import MCQDataset, collate_fn, compute_map3, OPTION_KEYS, set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_frac", type=float, default=None,
                    help="Fraction of train.csv to hold out for eval (default: from config.yaml)")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for the split (default: from config.yaml)")
    p.add_argument("--log_wandb", action="store_true",
                    help="Log the eval results to WandB")
    return p.parse_args()


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

    for batch in tqdm(val_loader, desc="Evaluating"):
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
        lora_path  = _cfg["model"]["save_path"],
        train_csv  = _cfg["data"]["train_csv"],
        max_length = _cfg["data"]["max_length"],
        val_frac   = args.val_frac if args.val_frac is not None else _cfg["data"]["val_frac"],
        seed       = args.seed if args.seed is not None else _cfg["seed"],
    )

    set_seed(CFG.seed)

    train_df = pd.read_csv(CFG.train_csv)
    val_df   = train_df.sample(frac=CFG.val_frac, random_state=CFG.seed).reset_index(drop=True)
    print(f"Loaded train.csv: {len(train_df)} rows | Held-out eval split: {len(val_df)} rows")

    model, tokenizer = load_model(CFG.model_name, CFG.lora_path)
    device = next(model.parameters()).device

    results = run_eval(model, tokenizer, val_df, CFG.max_length, batch_size=4, device=device)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Examples : {results['n_examples']}")
    print(f"  MAP@3    : {results['map3']:.4f}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print("=" * 40)

    if args.log_wandb:
        import wandb
        wandb.init(project=_cfg["wandb"]["project"], name=_cfg["wandb"]["run_name"] + "-eval")
        wandb.log({
            "eval/map3": results["map3"],
            "eval/accuracy": results["accuracy"],
            "eval/f1_macro": results["f1_macro"],
            "eval/n_examples": results["n_examples"],
        })
        wandb.finish()
