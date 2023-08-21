# Translate English to French API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/translate_en_to_fr`

Body:

```
{
    "text": "How old are you?"
}

```
