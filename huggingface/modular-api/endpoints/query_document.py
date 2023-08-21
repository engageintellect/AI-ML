from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel

document_qa = pipeline(model="impira/layoutlm-document-qa")

class DocumentQARequest(BaseModel):
    image_url: str
    question: str

def setup_query_document_api(app):
    @app.post("/api/query_document/")
    def query_document(data: DocumentQARequest):
        try:
            response = document_qa(image=data.image_url, question=data.question)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
