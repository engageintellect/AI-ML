from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import uvicorn


app = FastAPI()

# Pydantic model for request body
class TextForClassification(BaseModel):
    text: str
    candidate_labels: "List[str]"


# Initialize the classifier once, outside of the endpoint, for better performance
classifier = pipeline('zero-shot-classification')

@app.post("/")
def classify_text(request_data: TextForClassification):
    try:
        res = classifier(request_data.text, candidate_labels=request_data.candidate_labels)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
