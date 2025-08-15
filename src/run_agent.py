# src/run_agent.py
import os
import json
from llm_service import get_mistral_chat_response
from predict_intent import predict as predict_intent_func
from executor import execute_action
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
TOKENIZER_NAME = "distilbert-base-uncased"
CHECKPOINT_FOLDER = "checkpoint-30500" # Make sure this is the correct checkpoint number

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models", "nlu_classifier", CHECKPOINT_FOLDER))

def load_nlu_model():
    """Loads the NLU model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except OSError:
        print(f"Error: NLU model not found at {MODEL_PATH}")
        return None, None

def extract_slots_with_llm(user_command: str, intent: str) -> dict:
    """
    Uses the powerful Mistral LLM to extract slots from the user's command
    based on the predicted intent.
    """
    prompt = f"""
    Given the user's command and the predicted intent, extract the parameters (slots) and return them as a JSON object.
    
    User Command: "{user_command}"
    Predicted Intent: "{intent}"
    
    Example for 'os.app.open':
    Command: "can you open spotify for me" -> {{"app_name": "Spotify"}}
    
    Example for 'browser.open':
    Command: "bring up the website for the New York Times" -> {{"url": "nytimes.com"}}

    Example for 'browser.search':
    Command: "search for how to use git" -> {{"query": "how to use git"}}

    Now, extract the slots for the given command.
    IMPORTANT: Your response MUST be a single, valid JSON object and nothing else. Do not add any conversational text, explanations, or markdown formatting.
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response_text = get_mistral_chat_response(messages)
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_string = response_text[json_start:json_end]
            slots = json.loads(json_string)
            return slots
        else:
            print(f"Error: Could not find a valid JSON object in the LLM response: {response_text}")
            return {}

    except Exception as e:
        print(f"Error extracting slots with LLM: {e}")
        return {}


def main():
    """
    The main agent loop that wires everything together.
    """
    print("--- Loading Mini-JARVIS NLU Core ---")
    nlu_tokenizer, nlu_model = load_nlu_model()
    if not nlu_model:
        return
    print("--- NLU Core Loaded. Mini-JARVIS is ready. ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_command = input("You: ")
        if user_command.lower() in ["exit", "quit"]:
            break

        # 1. Predict Intent with the local NLU model
        predicted_intent, confidence = predict_intent_func(nlu_tokenizer, nlu_model, user_command)
        print(f"NLU -> Intent: {predicted_intent} (Confidence: {confidence:.2f})")

        # 2. Extract Slots with the powerful LLM
        print("LLM -> Extracting parameters...")
        slots = extract_slots_with_llm(user_command, predicted_intent)
        
        # --- MODIFICATION START ---
        # Check if the LLM returned a nested dictionary and flatten it if so.
        if predicted_intent in slots and isinstance(slots[predicted_intent], dict):
            print("LLM -> (Note: Unpacking nested dictionary from LLM response)")
            slots = slots[predicted_intent]
        # --- MODIFICATION END ---
            
        print(f"LLM -> Extracted Slots: {slots}")

        # 3. Execute the action
        execute_action(predicted_intent, slots)
        print("-" * 20)


if __name__ == "__main__":
    main()
