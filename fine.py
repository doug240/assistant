import os
import torch
import random
import datasets
import traceback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from torch.utils.data import DataLoader

from pathlib import Path
from datetime import datetime

data_dir = "D:/"  # Root directory to scan
model_name = "EleutherAI/gpt-neo-125M"

# ========== Helper Functions ==========
def find_parquet_files(directory):
    return list(Path(directory).rglob("*.parquet"))

def tokenize_function(example):
    return tokenizer(example["text"])

def try_load_dataset(file_path):
    try:
        dataset = datasets.load_dataset("parquet", data_files=file_path, split="train")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {file_path}\n{e}")
        return None

def preprocess_dataset(dataset, dataset_name):
    tokenized_path = f"D:/preprocessed/{dataset_name}"
    if os.path.exists(tokenized_path):
        return datasets.load_from_disk(tokenized_path)
    try:
        dataset = dataset.shuffle(seed=42)
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
        tokenized.save_to_disk(tokenized_path)
        return tokenized
    except Exception as e:
        print(f"Tokenization failed for {dataset_name}: {e}")
        return None

def train_model(dataset, dataset_name):
    output_dir = f"D:/trained_models/{dataset_name}"
    logging_dir = f"D:/logs/{dataset_name}"  

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=logging_dir,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        resume_from_checkpoint=os.path.exists(output_dir),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(min(100, len(dataset)))),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    try:
        trainer.train(resume_from_checkpoint=os.path.exists(output_dir))
        trainer.save_model(output_dir)
        print(f"\n✅ Finished training: {dataset_name}\n")
    except Exception as e:
        print(f"❌ Training failed for {dataset_name}: {e}\n{traceback.format_exc()}")


# ========== Main Process ==========
parquet_files = find_parquet_files(data_dir)
if not parquet_files:
    print("No datasets found.")
    exit()

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Loop through datasets
for pq_file in parquet_files:
    dataset_name = Path(pq_file).stem
    print(f"\n📦 Processing dataset: {dataset_name}")

    dataset = try_load_dataset(pq_file)
    if dataset is None:
        continue

    tokenized = preprocess_dataset(dataset, dataset_name)
    if tokenized is None:
        continue

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable()
        train_model(tokenized, dataset_name)
    except Exception as e:
        print(f"❌ Could not prepare model or train on {dataset_name}: {e}")
