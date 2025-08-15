# src/train_nlu.py
import transformers
print(f"--> Using transformers version: {transformers.__version__}")

import json
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import numpy as np

# --- CONFIGURATION ---
DATA_FILE = "../data/commands.jsonl"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../models/nlu_classifier"
NUM_EPOCHS = 3
BATCH_SIZE = 4

def main():
    """
    Main function to load data, train the NLU model, and save it.
    """
    print("--- Starting NLU Model Training ---")

    # 1. Load and prepare the dataset
    print(f"Loading dataset from {DATA_FILE}...")
    with open(DATA_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    # Create a mapping from intent names (strings) to integer labels
    unique_intents = sorted(list(set([item['intent'] for item in data])))
    intent2id = {intent: i for i, intent in enumerate(unique_intents)}
    id2intent = {i: intent for i, intent in enumerate(unique_intents)}
    
    print(f"Found {len(unique_intents)} unique intents.")

    # Add integer labels to our data
    for item in data:
        item['label'] = intent2id[item['intent']]

    # Check if stratification is possible.
    label_counts = Counter([d['label'] for d in data])
    min_samples_per_class = min(label_counts.values())
    stratify_param = [d['label'] for d in data] if min_samples_per_class >= 2 else None
    if stratify_param is None:
        print("\n--- WARNING: Stratification disabled. Smallest class has fewer than 2 samples. ---\n")

    # Split data into training and evaluation sets
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42, stratify=stratify_param)

    # Convert to Hugging Face Dataset objects
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'eval': Dataset.from_list(eval_data)
    })
    
    print("Dataset prepared and split.")

    # 2. Initialize Tokenizer and Model
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(unique_intents),
        id2label=id2intent,
        label2id=intent2id
    )

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

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
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
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
            logging_steps=10,
        )
    # --- MODIFICATION END ---

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
    )

    # 4. Train the model
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")
    
    print("\n--- Evaluating Final Model ---")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 5. Save the model and tokenizer
    print(f"Saving model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- Model Saved Successfully ---")


if __name__ == "__main__":
    main()
