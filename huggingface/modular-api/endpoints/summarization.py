from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel

class TextForSummarization(BaseModel):
    text: str
    min_length: int
    max_length: int

summarizer = pipeline("summarization")

def setup_summarization_api(app):
    @app.post("/api/summarize/")
    def summarize_text(request_data: TextForSummarization):
        try:
            res = summarizer(request_data.text, min_length=request_data.min_length, max_length=request_data.max_length)
            return res[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
