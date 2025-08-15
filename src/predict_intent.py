# src/predict_intent.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os # Import the os module

# --- CONFIGURATION ---
# --- MODIFICATION START ---
# We load the original tokenizer from the Hub, and our fine-tuned model from the local checkpoint.
TOKENIZER_NAME = "distilbert-base-uncased"
CHECKPOINT_FOLDER = "checkpoint-30500"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models", "nlu_classifier", CHECKPOINT_FOLDER))
# --- MODIFICATION END ---

def main():
    """
    Loads the fine-tuned NLU model and predicts intents for user input.
    """
    print(f"--- Loading NLU Intent Classifier ---")
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"Model Path: {MODEL_PATH}")
    
    # 1. Load the tokenizer and model from the saved directory
    try:
        # Load the standard tokenizer from the Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        # Load our fine-tuned model from the local checkpoint folder
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("--- Model Loaded Successfully ---")
    except OSError:
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please make sure the path is correct and the folder exists.")
        return

    print("\nEnter a command to test the intent classifier.")
    print("Type 'exit' or 'quit' to stop.\n")

    # 2. Start an interactive loop to get user input
    while True:
        text = input("You: ")
        if text.lower() in ["exit", "quit"]:
            break

        # 3. Predict the intent and confidence
        predicted_intent, confidence = predict(tokenizer, model, text)
        print(f"Predicted Intent: {predicted_intent} (Confidence: {confidence:.2f})")


def predict(tokenizer, model, text: str) -> tuple[str, float]:
    """
    Takes a text string and uses the model to predict the intent and confidence.
    
    Returns:
        A tuple containing the predicted intent name and the confidence score.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get model predictions (logits)
    with torch.no_grad(): # Disables gradient calculation for faster inference
        logits = model(**inputs).logits

    # Convert logits to probabilities using softmax
    probabilities = F.softmax(logits, dim=1)
    
    # Find the class with the highest probability
    confidence, predicted_class_id = torch.max(probabilities, dim=1)

    # Convert the class ID back to the intent name using the model's config
    predicted_intent = model.config.id2label[predicted_class_id.item()]
    
    return predicted_intent, confidence.item()


if __name__ == "__main__":
    main()
