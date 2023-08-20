# Sentiment API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/analyze_sentiment`

Body:

```
{
    "text": "I love using FastAPI!"
}
```
