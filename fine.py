import os
import torch
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
from pathlib import Path

data_dir = "D:/"  # Your datasets root
model_name = "EleutherAI/gpt-neo-125M"
fine_tuned_dir = "./fine_tuned_model"  # Save fine-tuned model here

def find_parquet_files(directory):
    return list(Path(directory).rglob("*.parquet"))

def tokenize_function(example):
    return tokenizer(example["text"])

def try_load_dataset(file_path):
    try:
        return datasets.load_dataset("parquet", data_files=file_path, split="train")
    except Exception:
        print(f"Failed to load dataset: {file_path}\n{traceback.format_exc()}")
        return None

def preprocess_dataset(dataset, dataset_name):
    tokenized_path = f"D:/preprocessed/{dataset_name}"
    if os.path.exists(tokenized_path):
        try:
            return datasets.load_from_disk(tokenized_path)
        except Exception:
            print(f"Failed to load tokenized dataset: {tokenized_path}\n{traceback.format_exc()}")
            return None
    try:
        dataset = dataset.shuffle(seed=42)
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
        tokenized.save_to_disk(tokenized_path)
        return tokenized
    except Exception:
        print(f"Tokenization failed for {dataset_name}:\n{traceback.format_exc()}")
        return None

def train_model(model, dataset, dataset_name):
    training_args = TrainingArguments(
        output_dir=fine_tuned_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"D:/logs/{dataset_name}",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
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
        trainer.train(resume_from_checkpoint=os.path.exists(os.path.join(fine_tuned_dir, "checkpoint-1")))
        trainer.save_model(fine_tuned_dir)
        tokenizer.save_pretrained(fine_tuned_dir)
        print(f"✅ Finished fine tuning: {dataset_name}")
    except Exception:
        print(f"Training failed for {dataset_name}:\n{traceback.format_exc()}")

parquet_files = find_parquet_files(data_dir)
if not parquet_files:
    print("⚠️ No datasets found.")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model_name)

for pq_file in parquet_files:
    dataset_name = Path(pq_file).stem
    print(f"Processing dataset: {dataset_name}")

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
        train_model(model, tokenized, dataset_name)
    except Exception:
        print(f"Could not train on {dataset_name}:\n{traceback.format_exc()}")

