# src/agent.py

# --- CHANGED HERE ---
# Import the new function name
from .llm_service import get_mistral_chat_response

class AuronAgent:
    def __init__(self, name="Auron"):
        self.name = name
        # Define the system prompt once, during initialization
        self.system_prompt = f"You are {self.name}, a helpful and concise AI assistant."

    def process_query(self, query: str) -> str:
        """
        Main AI logic — takes a user query, formats it for chat,
        and gets a response from Mistral.
        """
        # --- CHANGED HERE ---
        # Instead of creating one big string, create a list of message dictionaries
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Call the updated service function with the messages list
        return get_mistral_chat_response(messages)