from transformers import pipeline

en_fr_translator = pipeline("translation_en_to_fr")
res = en_fr_translator("How old are you?")

print(res)

