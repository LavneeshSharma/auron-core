# src/main.py
from .agent import AuronAgent

def main():
    print("🤖 Auron AI Agent (Mistral-7B-Instruct-v0.3)")
    print("Type 'exit' to quit.\n")

    agent = AuronAgent()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Auron: Goodbye!")
            break

        response = agent.process_query(user_input)
        print(f"Auron: {response}\n")

if __name__ == "__main__":
    main()
