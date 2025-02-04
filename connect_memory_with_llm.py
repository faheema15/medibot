import os
import re
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from spellchecker import SpellChecker

# Set up Hugging Face credentials
HF_TOKEN = "your_huggingface_token"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512},
        task="text-generation"
    )

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question in a detailed manner.
If you don’t know the answer, try to infer the closest relevant topic.
If the question is unclear or contains spelling errors, correct them before answering.
Do NOT make up facts. Provide a clear and structured answer.

Context: {context}
Question: {question}

Provide a well-structured response without listing questions beforehand.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Spell Checker
spell = SpellChecker()

def correct_query(query):
    words = query.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return " ".join(corrected_words)

# List of common greetings
greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

# Get user query
user_query = input("Write Query Here: ").strip().lower()
corrected_query = correct_query(user_query)

# Check if the user query is a greeting
if corrected_query in greetings:
    print("\nHi there! I'm Medibot. I'm here to assist with medical and health-related questions.")
else:
    response = qa_chain.invoke({'query': corrected_query})

    # Clean Markdown for Terminal
    clean_result = re.sub(r'\*\*(.*?)\*\*', r'\1', response.get("result", "I don't have enough data on that. If you have any concerns about your health, feel free to ask!"))
    clean_result = re.sub(r'^\s*Answer:\s*', '', clean_result, flags=re.IGNORECASE)  # Remove "Answer:" if it appears at the beginning


    # Print results in terminal
    print("\nRESULT: ", clean_result)
    print("\nSOURCE DOCUMENTS:")
    for doc in response.get("source_documents", []):
        print(f"- Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page_label', 'N/A')})")
