import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".gitignore")/".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


def get_pdf_text(pdf_paths):
    """Extracts text from a list of PDF file paths."""
    text = ""
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        for page in pages:
            text += page.page_content
    return text

def get_text_chunks(text):
    """Splits text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """Creates a conversational Question-Answering chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a
    wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    """Processes user input, runs the QA chain, and displays the response."""
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with your PDFs using Gemini Pro 💬")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)
    elif user_question:
        st.warning("Please upload and process your PDF files first.")

    with st.sidebar:
        st.title("Menu")
        st.write("Upload your PDF files and click 'Process' to start.")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing...This may take a moment."):
                    temp_dir = "temp_pdfs"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)

                    pdf_paths = []
                    for pdf in pdf_docs:
                        pdf_path = os.path.join(temp_dir, pdf.name)
                        with open(pdf_path, "wb") as f:
                            f.write(pdf.getbuffer())
                        pdf_paths.append(pdf_path)

                    raw_text = get_pdf_text(pdf_paths)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Processing Complete!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
