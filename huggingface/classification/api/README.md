# Classification API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/classification`

Body:

```
{
    "text": "This is a course about the Transformers library",
    "candidate_labels": ["education", "politics", "business"]
}

```
