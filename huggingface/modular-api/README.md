# HUGGINGFACE MODULAR API

## Description
This is a modular API that allows you to use the HuggingFace models in a simple way. It is based on the FastAPI framework and allows you to use the models in a simple way.

## Installation
```bash
pip install -r requirements.txt
sudo apt install tessaract-ocr
uvicorn main:app --reload
```

## Structure
```
fastapi_project/
│
├── main.py
│
├── endpoints/
│   ├── translation.py
│   ├── text_generation.py
│   ├── classification.py
│   ├── summarization.py
│   ├── sentiment_analysis.py
│   ├── ner.py
│   ├── sentence_embeddings.py
│   └── query_document.py
│
└── utils/
    └── helpers.py

```

## Usage
1. English to French Translation
2. Text Generation
3. Category Classification
4. Text Summarization
5. Sentiment Analysis
6. Named Entity Recognition (NER)
7. Sentance Transformers (Sentence Embeddings)
8. Document Quering


### 1. English to French Translation

---

**Endpoint**: `/api/translate_en_to_fr/`

**Method**: POST

**Payload**:

```json
{
  "text": "Hello world!"
}
```

**cURL**

```
curl -X 'POST' \
  'http://localhost:8000/api/translate_en_to_fr/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world!"}'
```

**Response**:

```json
{
  "translated_text": "Bonjour le monde!"
}
```

### 2. Text Generation

---

**Endpoint**: `/api/generate_text/`

**Method**: POST

**Payload**:

```json
{
  "text": "We are very happy",
  "max_length": 30,
  "num_return_sequences": 3
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/generate_text/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "We are very happy", "max_length": 30, "num_return_sequences": 3}'
```

**Response**:
A list of generated texts based on the input.

### 3. Category Classification

---

**Endpoint**: `/api/classification/`

**Method**: POST

**Payload**:

```json
{
  "text": "This is a course about the Transformers library.",
  "candidate_labels": ["education", "politics", "business"]
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/classification/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a course about the Transformers library.", "candidate_labels": ["education", "politics", "business"]}'

```

**Response**:
Predicted labels with their associated scores.

### 4. Text Summarization

---

**Endpoint**: `/api/summarize_text/`

**Method**: POST

**Payload**:

```json
{
  "text": "The artificial intelligence (AI) revolution is upon us...",
  "min_length": 5,
  "max_length": 20
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/summarize_text/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "The artificial intelligence (AI) revolution is upon us...", "min_length": 5, "max_length": 20}'

```

**Response**:
A summarized version of the input text.

### 5. Sentiment Analysis

---

**Endpoint**: `/api/analyze_sentiment/`

**Method**: POST

**Payload**:

```json
{
  "text": "The weather is amazing today!"
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/analyze_sentiment/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "The weather is amazing today!"}'

```

**Response**:
Predicted sentiment (positive, neutral, negative) with an associated score.

### 6. Named Entity Recognition (NER)

---

**Endpoint**: `/api/extract_entities/`

**Method**: POST

**Payload**:

```json
{
  "text": "My name is Jeff and I work in BTS with Emerging Technologies in Irvine, CA."
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/extract_entities/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "My name is Josh and I work at Emerging Technologies in Irvine, CA."}'

```

### 7. Sentence Transformers (Sentence Embeddings)

---

**Endpoint**: `/api/sentence_embeddings/`

**Method**: POST

**Payload**:

```json
{
  "sentences": ["this is a sentence", "this is another sentence"]
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/extract_entities/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": ["this is a sentence" , "this is another sentence"]}'

```

**Response**:
An array of sentence embeddings (tensors) for the input sentences.

### 8. Document Quering

---

**Endpoint**: `/api/query_document/`

**Method**: POST

**Payload**:

```json
{
  "image_url": "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
  "question": "Who is this shipping to?"
}
```

**cURL**:

```
curl -X 'POST' \
  'http://localhost:8000/api/query_document/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_url": "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
  "question": "Who is this shipping to?"
}
'

```

## Testing the API

You can test the API using various tools such as:

- The interactive FastAPI documentation at `http://localhost:8000/docs`
- Tools like `curl` or Postman
- Directly integrating it into your applications using appropriate HTTP client libraries.

Happy coding!
