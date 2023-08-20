from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
res = ner("My name is Josh and I work with Emerging Technologies in Irvine, CA.")

print(res)
