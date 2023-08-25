from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from pprint import pprint

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./saved_model_directory')
# After loading your saved tokenizer
print(tokenizer.convert_ids_to_tokens([100, 30522]))

model = AutoModelForTokenClassification.from_pretrained('./saved_model_directory')

# Create NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


def aggregate_entities(entities):
    agg_entities = []
    curr_agg_entity = None
    for ent in entities:
        if "##" not in ent['word']:
            if curr_agg_entity:
                agg_entities.append(curr_agg_entity)
            curr_agg_entity = ent
        else:
            curr_agg_entity['word'] += ent['word'].replace("##", "")
            curr_agg_entity['end'] = ent['end']
    if curr_agg_entity:
        agg_entities.append(curr_agg_entity)
    return agg_entities

# Example usage
example = "I am Josh and I work with the EA team in California."
result = nlp(example)

agg_result = aggregate_entities(result)
pprint(agg_result)
# print(result)
