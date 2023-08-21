from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel

class TextForSentiment(BaseModel):
    text: str

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def setup_sentiment_analysis_api(app):
    @app.post("/api/analyze_sentiment/")
    def analyze_sentiment(request_data: TextForSentiment):
        try:
            res = sentiment_analyzer(request_data.text)
            return res[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
