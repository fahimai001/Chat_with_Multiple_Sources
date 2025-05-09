import os
from typing import List, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate


def set_google_api_key():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


def load_document(file_path):
    """Loading document based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()

    loader = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
    }.get(file_extension)

    if not loader:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return loader(file_path).load()


def load_website(url):
    """Loading content from a website URL"""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    loader = WebBaseLoader(url)
    return loader.load()


def split_documents(documents, chunk_size = 1000, chunk_overlap = 200):
    """Spliting documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def create_vector_store(documents, persist_dir = "./chroma_db"):
    """Creating Chroma vector store from documents"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )


def build_prompt_template():
    """Building the prompt template for different content types"""
    template = """
    You are an AI assistant helping users with their questions about the content they've provided.
    The content may include documents (PDF, DOCX), text files, CSV data, or website content.
    
    For CSV data, provide structured answers with relevant statistics when appropriate.
    For website content, clearly reference the source website in your answers.
    
    {context}

    Question: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])


def setup_qa_chain():
    """Setting up the QA chain using Gemini model"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = build_prompt_template()
    return create_stuff_documents_chain(llm=model, prompt=prompt)