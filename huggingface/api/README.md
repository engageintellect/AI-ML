# Huggingface Transformers FastAPI

This FastAPI application provides endpoints for various NLP tasks using the Huggingface Transformers library. The current version includes:

1. English to French Translation
2. Text Generation
3. Zero-Shot Text Classification
4. Text Summarization
5. Sentiment Analysis

## Setting up the API

1. Make sure you have FastAPI and Uvicorn installed:

   ```
   pip install fastapi uvicorn
   ```

2. Install the Huggingface Transformers library:

   ```
   pip install transformers
   ```

3. Clone this repository and navigate to the directory containing the FastAPI application.

4. Start the server using:
   ```
   uvicorn main:app --reload
   ```

This will start the FastAPI app, making it accessible via `http://localhost:8000`.

## API Documentation & Usage

### 1. English to French Translation

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

### 3. Zero-Shot Text Classification

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

---

## Testing the API

You can test the API using various tools such as:

- The interactive FastAPI documentation at `http://localhost:8000/docs`
- Tools like `curl` or Postman
- Directly integrating it into your applications using appropriate HTTP client libraries.

Happy coding!
