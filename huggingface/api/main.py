from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModel, pipeline
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn.functional as F


from typing import List

app = FastAPI()

#################################################
# Translation API
#################################################
class TextForTranslation(BaseModel):
    text: str

en_fr_translator = pipeline("translation_en_to_fr")

@app.post("/api/translate_en_to_fr/")
def translate_en_to_fr(request_data: TextForTranslation):
    try:
        res = en_fr_translator(request_data.text)
        return {"translated_text": res[0]['translation_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#################################################
# Text Generation API
#################################################
class TextForGeneration(BaseModel):
    text: str
    max_length: int
    num_return_sequences: int

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

# Zero-Shot Classification API
class TextForClassification(BaseModel):
    text: str
    candidate_labels: List[str]

classifier = pipeline('zero-shot-classification')

@app.post("/api/classification/")
def classify_text(request_data: TextForClassification):
    try:
        res = classifier(request_data.text, candidate_labels=request_data.candidate_labels)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#################################################
# Summarization API
#################################################
class TextForSummarization(BaseModel):
    text: str
    min_length: int
    max_length: int

summarizer = pipeline("summarization")

@app.post("/api/summarize/")
def summarize_text(request_data: TextForSummarization):
    try:
        res = summarizer(request_data.text, min_length=request_data.min_length, max_length=request_data.max_length)
        return res[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#################################################
# Sentiment Analysis API
#################################################
class TextForSentiment(BaseModel):
    text: str

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/api/analyze_sentiment/")
def analyze_sentiment(request_data: TextForSentiment):
    try:
        res = sentiment_analyzer(request_data.text)
        return res[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#################################################
# NER API
#################################################
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


#################################################
# Sentence Embeddings API
#################################################
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel



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
