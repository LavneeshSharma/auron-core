# src/agent.py

# --- CHANGED HERE ---
# Import the new function name
from services.llm_service import get_mistral_chat_response
from .memory import Memory

class AuronAgent:
    def __init__(self, name="Auron"):
        self.name = name
        self.memory = Memory()
        self.system_prompt = f"You are {self.name}, a helpful and concise AI assistant."

    def process_query(self, query: str) -> str:
        """
        Main AI logic — takes a user query, formats it for chat,
        and gets a response from Mistral.
        """
        # --- CHANGED HERE ---
        last_intent, last_slots = self.memory.recall()
        
        # Build context-aware prompt
        context = ""
        if last_intent:
            context = f"Previous action: {last_intent} with {last_slots}\n"
        # Instead of creating one big string, create a list of message dictionaries
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Call the updated service function with the messages list
        return get_mistral_chat_response(messages)