import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# data loading steps
import pandas as pd
from langchain_community.document_loaders import TextLoader #pip install langchain-community
file_path = "faq.txt"  
loader = TextLoader(file_path)
data = loader.load() 
# split text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter #pip install langchain-text-splitters
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text=text_splitter.split_documents(data)
# embeddings
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings #pip install -qU langchain-huggingface #(requires sentence-transformers + torch)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Create vector store
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(documents=text, embedding=embeddings)
# retriever
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
#prompt template
from langchain_core.prompts import ChatPromptTemplate
system_prompt = """You are a helpful customer support chatbot.
Use the following context to answer the user's question as accurately as possible:
{context}
If the context does not contain the answer, respond with "I'm sorry, I don't have that information."""
prompt=ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")]) #from_messages(): Build a multi-role prompt easily
# llm 
from langchain_groq import ChatGroq #pip install langchain-groq
groq_api_key=os.getenv("groq_api_key")
llm=ChatGroq(groq_api_key=groq_api_key, model_name="groq/compound")
# chain creation #connecting your prompt, llm and vector_store together
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #pip install langchain-classic
from langchain_classic.chains.retrieval import create_retrieval_chain
question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever, question_answer_chain)
# chat history
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
contextualize_q_system_prompt = (
    "You are a fintech assistant. Given a chat history and the latest user question, "
    "reformulate the question into a standalone question that can be understood without the chat history. "
    "Do NOT answer the question here, just rewrite it clearly if needed.")
contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),])
history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
question_answer_chain=create_stuff_documents_chain(llm, qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever, question_answer_chain)
# Session-based Conversational RAG
from langchain_community.chat_message_histories import ChatMessageHistory #pip install langchain-community
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
