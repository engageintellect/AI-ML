from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Pydantic model for request body
class TextForGeneration(BaseModel):
    text: str
    max_length: int
    num_return_sequences: int

# Initialize the generator pipeline once, outside of the endpoint, for efficiency
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
