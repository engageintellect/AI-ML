from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Pydantic model for request body
class TextToSummarize(BaseModel):
    text: str
    min_length: int = 5
    max_length: int = 20

# Initialize the summarizer object once, outside of the endpoint, for better performance
summarizer = pipeline("summarization")

@app.post("/api/summarize/")
def get_summary(text_data: TextToSummarize):
    try:
        summary = summarizer(text_data.text, min_length=text_data.min_length, max_length=text_data.max_length)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

