from fastapi import HTTPException
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from utils.helpers import mean_pooling

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

class Sentences(BaseModel):
    sentences: list

def setup_sentence_embeddings_api(app):
    @app.post("/api/sentence_embeddings/")
    def get_sentence_embeddings(data: Sentences):
        try:
            encoded_input = tokenizer(data.sentences, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings_list = normalized_embeddings.tolist()
            return {"embeddings": embeddings_list}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
