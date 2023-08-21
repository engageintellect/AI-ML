from fastapi import FastAPI
from endpoints import translation, text_generation, classification, summarization, sentiment_analysis, ner, sentence_embeddings, query_document

app = FastAPI()

# Registering the endpoints
translation.setup_translation_api(app)
text_generation.setup_text_generation_api(app)
classification.setup_classification_api(app)
summarization.setup_summarization_api(app)
sentiment_analysis.setup_sentiment_analysis_api(app)
ner.setup_ner_api(app)
sentence_embeddings.setup_sentence_embeddings_api(app)
query_document.setup_query_document_api(app)
