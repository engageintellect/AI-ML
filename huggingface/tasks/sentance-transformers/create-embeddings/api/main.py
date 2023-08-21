from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

app = FastAPI()

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

class Sentences(BaseModel):
    sentences: list

@app.post("/api/sentence_embeddings/")
def get_sentence_embeddings(data: Sentences):
    try:
        # Tokenize sentences
        encoded_input = tokenizer(data.sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert tensor to list for a serializable response
        embeddings_list = normalized_embeddings.tolist()

        return {"embeddings": embeddings_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
