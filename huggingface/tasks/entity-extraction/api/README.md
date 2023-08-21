# Entity Extraction API

to run the api, run the following command:

```bash
uvicorn main:app --reload
```

## Usage

### Request

POST to `http://localhost:8000/api/extract_entities`

Body:

```
{
  "text": "My name is Jeff and I work in BTS with Emerging Technologies in Irvine, CA."
}
```
