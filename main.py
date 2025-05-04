from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.helper_func import (
    set_google_api_key,
    load_document,
    split_documents,
    create_vector_store,
    setup_qa_chain,
)

import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

set_google_api_key()
qa_chain = setup_qa_chain()
vector_store = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": "", "message": ""})


@app.post("/upload_query", response_class=HTMLResponse)
async def upload_and_query(
    request: Request,
    files: list[UploadFile] = File(...),
    question: str = Form(...)
):
    os.makedirs("temp_files", exist_ok=True)
    all_docs = []

    try:
        for file in files:
            file_path = f"temp_files/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            docs = load_document(file_path)
            all_docs.extend(docs)
            os.remove(file_path)

        chunks = split_documents(all_docs)
        global vector_store
        vector_store = create_vector_store(chunks)

        retriever = vector_store.as_retriever()
        relevant_docs = retriever.invoke(question)
        response = qa_chain.invoke({"context": relevant_docs, "question": question})
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": response,
            "message": "Files uploaded and processed."
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": "",
            "message": f"Error: {str(e)}"
        })
