# Sentence Transformers (Sentence Embeddings)

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
