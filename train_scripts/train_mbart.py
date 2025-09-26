import os
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from langdetect import detect

lang_code_map = {
    "hi": "hi_IN",
    "mr": "mr_IN",
    "sa": "sa_IN",
    "en": "en_XX",
}

def load_dataset_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    dataset_dict = {"text": lines, "target": lines}
    return Dataset.from_dict(dataset_dict)

def preprocess_function(examples, tokenizer):
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=96, truncation=True, padding="max_length")

    labels = []
    for text in examples["target"]:
        try:
            detected_lang = detect(text)
        except:
            detected_lang = "en"
        tgt_lang = lang_code_map.get(detected_lang, "en_XX")
        tokenizer.tgt_lang = tgt_lang
        label = tokenizer(text, max_length=96, truncation=True, padding="max_length")
        labels.append(label["input_ids"])

    model_inputs["labels"] = labels
    return model_inputs

def main():
    model_name = "facebook/mbart-large-50"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))

    print("Loading tokenizer and model...")
    tokenizer = MBartTokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Loading dataset...")
    dataset = load_dataset_from_file("multilingual_corpus.txt")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, batch_size=1000)

    training_args = TrainingArguments(
        output_dir="./mbart_finetuned",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        save_total_limit=2,
        fp16=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model("./mbart_finetuned")
    tokenizer.save_pretrained("./mbart_finetuned")

    print("Training complete!")

if __name__ == "__main__":
    main()
