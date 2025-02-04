import os
import re
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Define Hugging Face Credentials Once
HF_TOKEN = os.environ.get("HF_TOKEN")  # Load before using
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512},
        task="text-generation"
    )

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hola", "greetings", "howdy", "yo", "hi there"]
    return text.lower().strip() in greetings

def main():
    st.title("Ask Medibot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Get user input
    prompt = st.chat_input("Ask me a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Check if the user input is a greeting
        if is_greeting(prompt):
            greeting_response = "Hi there! I'm Medibot. I'm here to assist with medical and health-related questions."
            st.chat_message('assistant').markdown(greeting_response)
            st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
            return  # Stop further processing

        # Custom AI Response Processing
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical assistant AI. Answer the user's question in a clear, structured, and concise manner.
        - Do NOT return a list of questions.
        - Only provide a direct, relevant answer.
        - If the context does not contain enough information, say "I don’t know" rather than making up details.

        Context: {context}
        Question: {question}
        Answer:
        """



        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return  # Exit if FAISS DB is not loaded

            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            retrieved_docs = retriever.get_relevant_documents(prompt)  # Get retrieved documents

            # DEBUG: Print retrieved documents
            print("Retrieved Docs:", retrieved_docs)  # Check if it's retrieving meaningful content

            if not retrieved_docs:
                st.error("No relevant documents found. The knowledge base might be empty.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,  # Disable returning source documents
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            # DEBUG: Print model response
            print("Raw Response:", response)

            # Extract the result correctly
            if isinstance(response, dict) and "result" in response:
                result = response["result"]
            elif isinstance(response, str):
                result = response  # Some models return raw text directly
            else:
                result = "No relevant answer found."

            # Clean response
            cleaned_result = re.sub(r"\*\*(.*?)\*\*", r"\1", result)  # Remove Markdown bold
            cleaned_result = re.sub(r"^(Response:|Answer:)\s*", "", cleaned_result)  # Remove prefixes

            # Display response
            st.chat_message('assistant').markdown(cleaned_result.replace("\n", "<br>"), unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': cleaned_result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
