# train.py
"""
Minimal fine-tuning script using Hugging Face Trainer for causal LM.
Place plaintext corpora (one sentence per line or raw text) in settings.TRAINING_DATA_PATH
and adjust CORPUS_FILES in settings.py to point to them.

Run:
    python train.py --output_dir models/fine_tuned_small --epochs 1 --batch_size 2
"""
import argparse
from pathlib import Path
import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    model_source = str(settings.MODEL_PATH) if settings.MODEL_PATH else settings.MODEL_NAME
    logger.info("Loading tokenizer and model from %s", model_source)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
    except Exception:
        # try to use SentencePiece directly by pointing to file
        if Path(settings.TOKENIZER_MODEL_PATH).exists():
            # Note: converting SP model into HF tokenizer is preferred beforehand.
            raise RuntimeError("HF tokenizer not found. Please convert your SentencePiece into a HF tokenizer or provide a HF-compatible model.")
        else:
            raise

    model = AutoModelForCausalLM.from_pretrained(model_source)
    # Load dataset: combine all corpora files into one dataset (text)
    data_files = {}
    train_paths = []
    for lang, fname in settings.CORPUS_FILES.items():
        p = Path(settings.TRAINING_DATA_PATH) / fname
        if p.exists():
            train_paths.append(str(p))
    if not train_paths:
        raise RuntimeError("No training files found. Please place corpora in settings.TRAINING_DATA_PATH")

    ds = load_dataset("text", data_files={"train": train_paths})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        remove_unused_columns=False,
        fp16=True if torch.cuda.is_available() else False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized["train"]
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Fine-tuning complete. Saved to %s", args.output_dir)

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(settings.FINE_TUNED_MODEL_PATH), type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()
    main(args)
