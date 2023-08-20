from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize the NER pipeline
ner = pipeline("ner", grouped_entities=True)

# Pydantic model for request body
class TextForNER(BaseModel):
    text: str

@app.post("/api/extract_entities/")
async def extract_entities(request: TextForNER):
    try:
        result = ner(request.text)
        return {"entities": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
