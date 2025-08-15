# src/generate_dataset.py
import json
import random
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_FILE = "../data/commands.jsonl"
NUM_SAMPLES = 100000

# --- DATA FOR GENERATION (EXPANDED) ---
APP_NAMES = ["Chrome", "Spotify", "VS Code", "Terminal", "Calculator", "Photoshop", "Netflix", "Discord", "Slack", "Firefox", "Word", "Excel", "Brave Browser"]
FILE_NAMES = ["budget report", "presentation slides", "project plan", "meeting notes", "my resume", "vacation photos", "quarterly earnings", "user data"]
FILE_TYPES = ["document", "spreadsheet", "presentation", "image", "pdf", "text file", "archive", "script"]
URLS = ["google.com", "youtube.com", "github.com", "huggingface.co", "wikipedia.org", "reddit.com", "amazon.com", "linkedin.com", "nytimes.com"]
SEARCH_QUERIES = [
    "the weather in Howrah", "the latest news on AI", "how to bake a cake", "python tutorials", 
    "the best restaurants near me", "what is a large language model", "the history of India", "how to use git", "machine learning courses"
]
NOTE_CONTENTS = [
    "pick up laundry tomorrow at 5pm", "call mom this evening", "buy milk, eggs, and bread",
    "the meeting is rescheduled to 3 PM", "follow up on the project status", "remember to submit the report by Friday"
]

# --- INTENT TEMPLATES (EXPANDED FOR MORE VARIETY) ---
INTENT_DEFINITIONS = {
    "os.app.open": {
        "templates": [
            "open {app_name}", "launch {app_name}", "start {app_name}", "can you open {app_name} for me?",
            "I need to use {app_name}", "run {app_name}", "fire up {app_name}", "get {app_name} running"
        ],
        "slots": {"app_name": APP_NAMES}
    },
    "os.app.close": {
        "templates": [
            "close {app_name}", "shut down {app_name}", "quit {app_name}", "exit out of {app_name}",
            "I'm done with {app_name}, please close it", "terminate {app_name}"
        ],
        "slots": {"app_name": APP_NAMES}
    },
    "file.search": {
        "templates": [
            "find my {file_name}", "search for a file called {file_name}", "look for my {file_name} {file_type}",
            "show me all {file_type}s", "where is my {file_name}?", "I need to find the {file_name} {file_type}"
        ],
        "slots": {"file_name": FILE_NAMES, "file_type": FILE_TYPES}
    },
    "browser.open": {
        "templates": [
            "go to {url}", "open the website {url}", "take me to {url}", "navigate to {url}",
            "bring up the website for {url}", "I want to visit {url}"
        ],
        "slots": {"url": URLS}
    },
    "browser.search": {
        "templates": [
            "search for {query}", "look up {query}", "find information about {query}", "what is {query}?",
            "can you find out {query}", "google {query}"
        ],
        "slots": {"query": SEARCH_QUERIES}
    },
    "productivity.note.create": {
        "templates": [
            "create a note that says {content}", "remind me to {content}", "take a note: {content}",
            "new note: {content}", "jot down that {content}", "make a memo about {content}"
        ],
        "slots": {"content": NOTE_CONTENTS}
    }
}

# (The rest of the script remains the same)

def generate_command():
    intent_name = random.choice(list(INTENT_DEFINITIONS.keys()))
    intent_data = INTENT_DEFINITIONS[intent_name]
    template = random.choice(intent_data["templates"])
    slots = {}
    final_text = template
    placeholders = [p.strip('{}') for p in template.split() if p.startswith('{')]
    for placeholder in placeholders:
        possible_values = intent_data["slots"].get(placeholder)
        if possible_values:
            chosen_value = random.choice(possible_values)
            slots[placeholder] = chosen_value
            final_text = final_text.replace(f"{{{placeholder}}}", chosen_value, 1)
    return {"text": final_text, "intent": intent_name, "slots": slots}

def main():
    print(f"--- Starting dataset generation for {NUM_SAMPLES} samples ---")
    with open(OUTPUT_FILE, 'w') as f:
        for _ in tqdm(range(NUM_SAMPLES), desc="Generating Commands"):
            entry = generate_command()
            f.write(json.dumps(entry) + '\n')
    print(f"\n--- Successfully generated and saved dataset to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()
