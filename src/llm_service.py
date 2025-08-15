# src/llm_service.py
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env automatically
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Please set HF_API_TOKEN in your environment variables.")

# Initialize Hugging Face client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HF_API_TOKEN
)

# --- CHANGED FROM HERE ---

def get_mistral_chat_response(messages: list) -> str:
    """
    Send a list of messages to Mistral-7B-Instruct-v0.3 for chat completion.
    """
    try:
        # Use the chat_completion method instead of text_generation
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        # The response is an object, so we need to extract the message content
        return response.choices[0].message.content
    except Exception as e:
        # Check for the specific error you were seeing before
        if "is not supported for task text-generation" in str(e):
            return "API Error: The model is correctly called for chat, but the API endpoint may have issues. Please check Hugging Face's status."
        return f"Error from Mistral API: {e}"

# --- TO HERE ---