# src/run_agent.py
import os
import json
import re

# --- Clean, package-based imports ---
from services.llm_service import get_mistral_chat_response
from nlu.classifier import load_nlu_model, predict_intent
from skills.executor import execute_action
from agent.memory import Memory

# --- CONFIGURATION ---
TOKENIZER_NAME = "distilbert-base-uncased"
# IMPORTANT: Update this to the checkpoint number of your newly trained model
CHECKPOINT_FOLDER = "checkpoint-30000" 

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models", "nlu_classifier", CHECKPOINT_FOLDER))

def extract_slots_with_llm(user_command: str, intent: str) -> dict:
    """
    Extracts slots from user command using LLM with robust fallbacks.
    """
    prompt = f"""
    Extract the parameters (slots) for the predicted intent strictly as JSON.

    User Command: "{user_command}"
    Predicted Intent: "{intent}"

    ✅ Example JSON Response:
    {{
      "media_title": "Shape of You",
      "platform": "YouTube"
    }}

    ❌ Do not include explanations, text, or markdown. Only valid JSON.
    """
    messages = [{"role": "user", "content": prompt}]
    slots = {}

    try:
        response_text = get_mistral_chat_response(messages).strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_string = response_text[json_start:json_end]
            slots = json.loads(json_string)
    except Exception as e:
        print(f"LLM JSON parsing failed: {e}. Relying on fallbacks.")

    # --- Fallback 1: Regex on LLM's conversational response ---
    if not slots or all(v == "" for v in slots.values()):
        mt_match = re.search(r"Media Title:\s*['\"]?([^'\"]+)['\"]?", response_text, re.I)
        pf_match = re.search(r"Platform:\s*['\"]?([^'\"]+)['\"]?", response_text, re.I)
        if mt_match:
            slots["media_title"] = mt_match.group(1).strip()
        if pf_match:
            slots["platform"] = pf_match.group(1).strip()

    # --- Fallback 2: Direct regex on user command ---
    if intent == "media.play" and not slots.get("media_title"):
        match = re.match(r"play (.+?)(?: on (\w+))?$", user_command, re.I)
        if match:
            slots["media_title"] = match.group(1).strip()
            if match.group(2):
                slots["platform"] = match.group(2).capitalize()

    if intent == "os.app.open" and not slots.get("app_name"):
        match = re.match(r"(?:open|launch|start) (.+)", user_command, re.I)
        if match:
            slots["app_name"] = match.group(1).strip()

    if intent == "browser.search" and not slots.get("query"):
        match = re.match(r"(?:search|google) (.+)", user_command, re.I)
        if match:
            slots["query"] = match.group(1).strip()

    # --- Fallback 3: Keyword cleanup ---
    if intent == "media.play" and not slots.get("media_title"):
        cleaned = re.sub(r"\b(play|on youtube|on spotify|song|video)\b", "", user_command, flags=re.I).strip()
        slots["media_title"] = cleaned
        if "youtube" in user_command.lower():
            slots["platform"] = "YouTube"
        elif "spotify" in user_command.lower():
            slots["platform"] = "Spotify"

    if intent == "productivity.note.create" and not slots.get("content"):
        match = re.search(r"(?:create note|make note|note)\s+(.+)", user_command, re.I)
        if match:
            slots["content"] = match.group(1).strip()

    return slots

def normalize_slots(intent: str, slots: dict) -> dict:
    """
    Normalizes slot names so the executor gets a consistent format.
    """
    normalized_slots = {}
    if intent == "media.play":
        normalized_slots["media_title"] = (slots.get("media_title") or slots.get("song_title") or slots.get("song_name") or slots.get("track") or "")
        normalized_slots["platform"] = (slots.get("platform") or slots.get("source") or slots.get("service") or "YouTube")
    elif intent == "browser.open":
        normalized_slots["url"] = slots.get("url") or slots.get("website") or ""
    elif intent == "browser.search":
        normalized_slots["query"] = slots.get("query") or slots.get("search") or ""
    elif intent == "os.app.open":
        normalized_slots["app_name"] = slots.get("app_name") or slots.get("application") or ""
    elif intent == "os.app.close":
        normalized_slots["app_name"] = slots.get("app_name") or slots.get("application") or ""
    elif intent == "productivity.note.create":
        normalized_slots["content"] = (slots.get("content") or slots.get("note") or slots.get("text") or "")
    else:
        normalized_slots = slots
    return normalized_slots

def main():
    """
    The main agent loop that wires everything together.
    """
    print("--- Loading Mini-JARVIS NLU Core ---")
    nlu_tokenizer, nlu_model = load_nlu_model(MODEL_PATH, TOKENIZER_NAME)
    if not nlu_model:
        return
        
    agent_memory = Memory()
    print("--- NLU Core Loaded. Mini-JARVIS is ready. ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_command = input("You: ")
        if user_command.lower() in ["exit", "quit"]:
            break

        # 1. Predict Intent
        predicted_intent, confidence = predict_intent(nlu_tokenizer, nlu_model, user_command)
        print(f"NLU -> Intent: {predicted_intent} (Confidence: {confidence:.2f})")
        
        if confidence < 0.6:
            print(f"⚠️  I'm not sure what you mean. Could you rephrase that?")
            continue

        # 2. Extract & Normalize Slots
        print("LLM -> Extracting parameters...")
        raw_slots = extract_slots_with_llm(user_command, predicted_intent)
        slots = normalize_slots(predicted_intent, raw_slots)
        print(f"LLM -> Normalized Slots: {slots}")

        # 3. Update Memory
        agent_memory.update(predicted_intent, slots)

        # 4. Execute Action
        execute_action(predicted_intent, slots)
        print("-" * 20)

if __name__ == "__main__":
    main()
