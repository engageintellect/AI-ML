from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel
from utils.helpers import numpy_to_python

ner = pipeline("ner", grouped_entities=True)

class TextForNER(BaseModel):
    text: str

def setup_ner_api(app):
    @app.post("/api/extract_entities/")
    async def extract_entities(request: TextForNER):
        try:
            result = ner(request.text)
            cleaned_result = numpy_to_python(result)
            return {"entities": cleaned_result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
