# Summarization API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/summarize`

Body:

```
{
    "text": "The artificial intelligence (AI) revolution is upon us, and companies must prepare to adapt to this change. ...",
    "min_length": 5,
    "max_length": 20
}

```
