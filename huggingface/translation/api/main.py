from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Pydantic model for request body
class TextForTranslation(BaseModel):
    text: str

# Initialize the translation pipeline once, outside of the endpoint, for better performance
en_fr_translator = pipeline("translation_en_to_fr")

@app.post("/api/translate_en_to_fr/")
def translate_en_to_fr(request_data: TextForTranslation):
    try:
        res = en_fr_translator(request_data.text)
        return {"translated_text": res[0]['translation_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
