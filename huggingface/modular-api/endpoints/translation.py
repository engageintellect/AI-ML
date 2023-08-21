from fastapi import HTTPException
from transformers import pipeline
from pydantic import BaseModel

class TextForTranslation(BaseModel):
    text: str

en_fr_translator = pipeline("translation_en_to_fr")

def setup_translation_api(app):
    @app.post("/api/translate_en_to_fr/")
    def translate_en_to_fr(request_data: TextForTranslation):
        try:
            res = en_fr_translator(request_data.text)
            return {"translated_text": res[0]['translation_text']}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
