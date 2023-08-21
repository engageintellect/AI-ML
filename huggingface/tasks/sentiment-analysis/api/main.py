from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Pydantic model for request body
class TextForAnalysis(BaseModel):
    text: str

# Initialize the classifier once, outside of the endpoint, for better performance
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/api/analyze_sentiment/")
def analyze_sentiment(text_data: TextForAnalysis):
    try:
        res = classifier(text_data.text)
        return res[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
