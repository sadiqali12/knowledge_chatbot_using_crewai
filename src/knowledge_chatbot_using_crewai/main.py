from mem0 import MemoryClient
from knowledge_chatbot_using_crewai.crew import CrewaiKnowledgeChatbot

client = MemoryClient()

def run():
    history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        chat_history = "\\n".join(history)

        inputs = {
            "user_message": f"{user_input}",
            "history": f"{chat_history}",
        }

        response = CrewaiKnowledgeChatbot().crew().kickoff(inputs=inputs)

        history.append(f"User: {user_input}")
        history.append(f"Assistant: {response}")
        client.add(user_input, user_id="User")

        print(f"Assistant: {response}")