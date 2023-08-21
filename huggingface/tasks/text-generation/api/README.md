# Text Generation API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/generate_text`

Body:

```
{
    "text": "We are very happy to show you the 🤗 Transformers library.",
    "max_length": 30,
    "num_return_sequences": 5
}
```
