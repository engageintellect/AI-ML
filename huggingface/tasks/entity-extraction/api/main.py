from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np

app = FastAPI()

# Initialize the NER pipeline
ner = pipeline("ner", grouped_entities=True)

# Pydantic model for request body
class TextForNER(BaseModel):
    text: str

def numpy_to_python(data):
    if isinstance(data, (np.ndarray, np.generic)):
        return numpy_to_python(data.tolist())
    elif isinstance(data, list):
        return [numpy_to_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: numpy_to_python(value) for key, value in data.items()}
    else:
        return data

@app.post("/api/extract_entities/")
async def extract_entities(request: TextForNER):
    try:
        result = ner(request.text)
        cleaned_result = numpy_to_python(result)
        return {"entities": cleaned_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
