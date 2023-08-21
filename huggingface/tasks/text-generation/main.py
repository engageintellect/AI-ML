from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

res = generator(
    'We are very happy to show you the 🤗 Transformers library.',
    max_length=30,
    num_return_sequences=5,
)	

print(res)