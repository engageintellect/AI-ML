from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./saved_model_directory')
# After loading your saved tokenizer
print(tokenizer.convert_ids_to_tokens([100, 30522]))

model = AutoModelForTokenClassification.from_pretrained('./saved_model_directory')

# Create NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Example usage
example = "I am josh, I work with BTS in California."
result = nlp(example)
print(result)
