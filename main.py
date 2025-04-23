from fastapi import FastAPI, UploadFile, File

from src.helper_func import (set_google_api_key,
                             load_document,
                             split_documents,
                            create_vector_store,
                            setup_qa_chain
)


import shutil
import os

app = FastAPI()
set_google_api_key()
qa_chain = setup_qa_chain()
vector_store = None

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_files/{file.filename}"
    os.makedirs("temp_files", exist_ok=True)

    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        documents = load_document(file_path)
        chunks = split_documents(documents)
        global vector_store
        vector_store = create_vector_store(chunks)
        return {"message": "File processed and vector store created."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(file_path)


@app.get("/query/")
async def query(question: str):
    if not vector_store:
        return {"error": "Vector store not initialized. Upload a document first."}


    retriever = vector_store.as_retriever()
    relevant_docs = retriever.invoke(question)
    response = qa_chain.invoke({"context": relevant_docs, "question": question})
    return {"answer": response}
