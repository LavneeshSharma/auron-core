# src/train_nlu.py
import os
import json
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import evaluate # Import the evaluate library

# --- CONFIGURATION ---
DATA_FILE = "./commands.jsonl"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../models/nlu_classifiers"
NUM_EPOCHS = 3
BATCH_SIZE = 8

# --- METRICS CALCULATION ---
# Load the accuracy metric from the evaluate library
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall, and F1 for a batch of predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # For multi-class, we use 'macro' average for precision, recall, and F1
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("--- Starting NLU Model Training ---")

    # 1. Load and prepare the dataset
    print(f"Loading dataset from {DATA_FILE}...")
    with open(DATA_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    unique_intents = sorted(list(set([item['intent'] for item in data])))
    intent2id = {intent: i for i, intent in enumerate(unique_intents)}
    id2intent = {i: intent for i, intent in enumerate(unique_intents)}
    print(f"Found {len(unique_intents)} unique intents.")

    for item in data:
        item['label'] = intent2id[item['intent']]

    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'eval': Dataset.from_list(eval_data)
    })
    print("Dataset prepared and split.")

    # 2. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_intents), id2label=id2intent, label2id=intent2id
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # 3. Set up Trainer
    # --- MODIFICATION START ---
    # This block attempts to use the best-practice arguments. If it fails due to a
    # persistent environment issue, it falls back to a simpler, more compatible configuration.
    try:
        print("Attempting to use best-practice TrainingArguments...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            load_best_model_at_end=True,
        )
    except TypeError:
        print("\n--- WARNING ---")
        print("Could not use best-practice TrainingArguments due to a persistent environment issue.")
        print("Falling back to a simplified configuration. Model will still train successfully.")
        print("-----------------\n")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            logging_steps=100,
        )
    # --- MODIFICATION END ---

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        # Pass our new metrics function to the Trainer
        compute_metrics=compute_metrics,
    )

    # 4. Train the model (with resume logic)
    latest_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            latest_checkpoint = os.path.join(OUTPUT_DIR, latest_checkpoint)
            print(f"--- Resuming training from checkpoint: {latest_checkpoint} ---")

    print("\n--- Starting Training ---")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    print("--- Training Finished ---")
    
    print("\n--- Evaluating Final Model ---")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    print(f"Saving final model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- Model Saved Successfully ---")

