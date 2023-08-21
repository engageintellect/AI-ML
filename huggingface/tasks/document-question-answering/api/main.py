#################################################
# Make sure to install the following dependencies:
# $ sudo apt install tesseract-ocr
#################################################

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize the document_qa pipeline
document_qa = pipeline(model="impira/layoutlm-document-qa")

class DocumentQARequest(BaseModel):
    image_url: str
    question: str

@app.post("/api/query_document/")
def query_document(data: DocumentQARequest):
    try:
        response = document_qa(image=data.image_url, question=data.question)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

