# terminal_chat.py
from app import conversational_rag_chain 
import uuid
# Generate a unique session ID for chat history
SESSION_ID = str(uuid.uuid4())
print("Welcome to the FAQ Chatbot! Type 'exit' to quit.\n")
while True:
    # Read user input
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    if not user_input:
        continue  # skip empty input
    # Call your conversational RAG chain
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": SESSION_ID}}
    )
    # Print bot's response
    print("Bot:", response["answer"])