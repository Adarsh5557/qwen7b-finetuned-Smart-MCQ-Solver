"""
Inference-only script — runs the saved LoRA checkpoint on test.csv
and writes a submission file. Does not require retraining.

Usage:
    python test.py
"""

import os
import yaml
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OPTION_KEYS = ["A", "B", "C", "D", "E"]


# ─────────────────────────────────────────────────────────
# CONFIG LOADING (from config.yaml)
# ─────────────────────────────────────────────────────────

with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

CFG = SimpleNamespace(
    model_name     = _cfg["model"]["name"],
    lora_path      = _cfg["model"]["save_path"],
    test_csv       = _cfg["data"]["test_csv"],
    submission_csv = _cfg["data"]["submission_csv"],
    max_length     = _cfg["data"]["max_length"],
    batch_size     = 2,  # inference only, no backward pass — safe to keep small
)

SYSTEM_PROMPT = (
    "You are an expert at answering multiple choice questions. "
    "Given a question and five options, identify the most likely correct answer. "
    "Respond with only a single letter: A, B, C, D, or E."
)


def build_prompt(prompt_text: str, options: dict) -> str:
    option_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt_text}\n\n"
        f"Options:\n{option_str}\n\n"
        f"Which option is correct? Answer with a single letter.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class MCQDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        options = {k: str(row[k]) for k in OPTION_KEYS}
        prompt  = build_prompt(str(row["prompt"]), options)
        enc     = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def collate_fn(batch, pad_token_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    ids, masks = [], []
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        ids.append(torch.cat([
            torch.full((pad,), pad_token_id, dtype=torch.long),
            x["input_ids"],
        ]))
        masks.append(torch.cat([
            torch.zeros(pad, dtype=torch.long),
            x["attention_mask"],
        ]))
    return {
        "input_ids":      torch.stack(ids),
        "attention_mask": torch.stack(masks),
    }


def load_model(model_name, lora_path):
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print("Model ready.")
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
def predict(model, tokenizer, test_df):
    device           = next(model.parameters()).device
    option_token_ids = get_option_token_ids(tokenizer).to(device)
    pad_id           = tokenizer.pad_token_id

    dataset = MCQDataset(test_df, tokenizer, CFG.max_length)
    loader  = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
    )

    all_top3 = []

    for batch in tqdm(loader, desc="Predicting"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
        )
        last_logits   = outputs.logits[:, -1, :].float()
        last_logits   = torch.nan_to_num(last_logits, nan=0.0)
        option_logits = last_logits[:, option_token_ids]

        top3 = torch.argsort(option_logits, dim=-1, descending=True)[:, :3]
        for row in top3.cpu().numpy():
            all_top3.append(" ".join(OPTION_KEYS[i] for i in row))

    return pd.DataFrame({
        "ID":         test_df["id"].values,
        "Prediction": all_top3,
    })


if __name__ == "__main__":
    test_df = pd.read_csv(CFG.test_csv)
    print(f"Test: {len(test_df)} rows")

    model, tokenizer = load_model(CFG.model_name, CFG.lora_path)

    submission = predict(model, tokenizer, test_df)
    submission.to_csv(CFG.submission_csv, index=False)

    print(f"\nSaved → {CFG.submission_csv}")
    print(submission.head(10).to_string(index=False))
