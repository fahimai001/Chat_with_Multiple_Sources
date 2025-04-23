# Document Q&A API

A FastAPI application that enables question answering against uploaded documents using vector search and large language models.

## Features

- Upload documents in various formats (PDF, DOCX, TXT, CSV)
- Process and chunk documents automatically
- Create vector embeddings for semantic search
- Query your documents using natural language
- Receive contextually relevant answers to your questions

## How It Works

1. Documents are uploaded through the API
2. Text is extracted, processed, and split into chunks
3. Vector embeddings are generated for each chunk
4. When a query is submitted, the most relevant document chunks are retrieved
5. A language model uses the context from these chunks to generate an answer

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - fastapi
  - uvicorn
  - python-multipart
  - langchain
  - sentence-transformers

### Installation

```bash
# Clone the repository
git clone https://github.com/fahimai001/Chat_with_Multiple_Sources
cd document-qa-api

# Install dependencies
pip install -r requirements.txt

# Set up your Google API key for the language model
export GOOGLE_API_KEY=your_api_key_here
```

### Usage

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Upload a document:
```bash
curl -X POST -F "file=@path/to/your/document.pdf" http://localhost:8000/upload/
```

3. Query your document:
```bash
curl "http://localhost:8000/query/?question=What%20is%20the%20main%20topic%20of%20the%20document?"
```

## API Endpoints

- `POST /upload/`: Upload a document for processing
- `GET /query/?question=your_question`: Ask questions about uploaded documents

## Project Structure

```
Chat_with_Multiple_Sources/
├── main.py               # FastAPI application
├── src/
│   └── helper_func.py    # Utility functions for document processing and Q&A
├── temp_files/           # Temporary storage for uploaded files
└── requirements.txt      # Project dependencies
```