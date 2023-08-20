from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List

app = FastAPI()

# Translation API
class TextForTranslation(BaseModel):
    text: str

en_fr_translator = pipeline("translation_en_to_fr")

@app.post("/api/translate_en_to_fr/")
def translate_en_to_fr(request_data: TextForTranslation):
    try:
        res = en_fr_translator(request_data.text)
        return {"translated_text": res[0]['translation_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text Generation API
class TextForGeneration(BaseModel):
    text: str
    max_length: int
    num_return_sequences: int

generator = pipeline('text-generation', model='distilgpt2')

@app.post("/api/generate_text/")
def generate_text(request_data: TextForGeneration):
    try:
        res = generator(
            request_data.text,
            max_length=request_data.max_length,
            num_return_sequences=request_data.num_return_sequences
        )
        return {"generated_texts": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Zero-Shot Classification API
class TextForClassification(BaseModel):
    text: str
    candidate_labels: List[str]

classifier = pipeline('zero-shot-classification')

@app.post("/api/classification/")
def classify_text(request_data: TextForClassification):
    try:
        res = classifier(request_data.text, candidate_labels=request_data.candidate_labels)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization API
class TextForSummarization(BaseModel):
    text: str
    min_length: int
    max_length: int

summarizer = pipeline("summarization")

@app.post("/api/summarize/")
def summarize_text(request_data: TextForSummarization):
    try:
        res = summarizer(request_data.text, min_length=request_data.min_length, max_length=request_data.max_length)
        return res[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment Analysis API
class TextForSentiment(BaseModel):
    text: str

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/api/analyze_sentiment/")
def analyze_sentiment(request_data: TextForSentiment):
    try:
        res = sentiment_analyzer(request_data.text)
        return res[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NER API
class TextForNER(BaseModel):
    text: str

ner = pipeline("ner", grouped_entities=True)

@app.post("/api/extract_entities/")
def extract_entities(request_data: TextForNER):
    try:
        res = ner(request_data.text)
        return {"entities": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
