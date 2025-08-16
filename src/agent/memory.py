# src/memory.py

class Memory:
    def __init__(self):
        # Keeps track of last few interactions
        self.last_intent = None
        self.last_slots = {}

    def update(self, intent: str, slots: dict):
        """Update memory with latest intent and slots."""
        self.last_intent = intent
        self.last_slots = slots

    def recall(self):
        """Return last remembered intent and slots."""
        return self.last_intent, self.last_slots

    def clear(self):
        """Clear memory."""
        self.last_intent = None
        self.last_slots = {}
