from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel
from typing import List

class TextForClassification(BaseModel):
    text: str
    candidate_labels: List[str]

classifier = pipeline('zero-shot-classification')

def setup_classification_api(app):
    @app.post("/api/classification/")
    def classify_text(request_data: TextForClassification):
        try:
            res = classifier(request_data.text, candidate_labels=request_data.candidate_labels)
            return res
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
