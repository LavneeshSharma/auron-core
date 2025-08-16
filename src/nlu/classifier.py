# src/nlu/classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

def load_nlu_model(model_path: str, tokenizer_name: str):
    """
    Loads the fine-tuned NLU model from a local path and the corresponding
    tokenizer from the Hugging Face Hub.
    """
    print(f"--- Loading NLU Intent Classifier ---")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Model Path: {model_path}")
    
    try:
        # Load the standard tokenizer from the Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Load our fine-tuned model from the local checkpoint folder
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("--- NLU Model Loaded Successfully ---")
        return tokenizer, model
    except OSError:
        print(f"Error: NLU model not found at {model_path}")
        print("Please make sure the path is correct and the folder exists.")
        return None, None

def predict_intent(tokenizer, model, text: str) -> tuple[str, float]:
    """
    Takes a text string and uses the model to predict the intent and confidence.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = F.softmax(logits, dim=1)
    confidence, predicted_class_id = torch.max(probabilities, dim=1)
    predicted_intent = model.config.id2label[predicted_class_id.item()]
    
    return predicted_intent, confidence.item()
