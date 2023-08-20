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

**Endpoint**: `/translate_en_to_fr/`

**Method**: POST

**Payload**:

```json
{
  "text": "Hello world!"
}
```

**Response**:

```json
{
  "translated_text": "Bonjour le monde!"
}
```

### 2. Text Generation

**Endpoint**: `/generate_text/`

**Method**: POST

**Payload**:

```json
{
  "text": "We are very happy",
  "max_length": 30,
  "num_return_sequences": 3
}
```

**Response**:
A list of generated texts based on the input.

### 3. Zero-Shot Text Classification

**Endpoint**: `/classify_text/`

**Method**: POST

**Payload**:

```json
{
  "text": "This is a course about the Transformers library.",
  "candidate_labels": ["education", "politics", "business"]
}
```

**Response**:
Predicted labels with their associated scores.

### 4. Text Summarization

**Endpoint**: `/summarize_text/`

**Method**: POST

**Payload**:

```json
{
  "text": "The artificial intelligence (AI) revolution is upon us...",
  "min_length": 5,
  "max_length": 20
}
```

**Response**:
A summarized version of the input text.

### 5. Sentiment Analysis

**Endpoint**: `/analyze_sentiment/`

**Method**: POST

**Payload**:

```json
{
  "text": "The weather is amazing today!"
}
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
