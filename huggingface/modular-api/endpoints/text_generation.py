from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel

class TextForGeneration(BaseModel):
    text: str
    max_length: int
    num_return_sequences: int

generator = pipeline('text-generation', model='distilgpt2')

def setup_text_generation_api(app):
    @app.post("/api/generate_text/")
    def generate_text(request_data: TextForGeneration):
        try:
            res = generator(request_data.text, max_length=request_data.max_length, num_return_sequences=request_data.num_return_sequences)
            return {"generated_texts": res}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
