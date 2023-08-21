import argparse
from transformers import pipeline

# Create the parser
parser = argparse.ArgumentParser(description="Perform sentiment analysis")

# Add an argument
parser.add_argument('Text', type=str, help="the text to analyze")

# Parse the argument
args = parser.parse_args()

# Load the classifier
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Perform sentiment analysis
res = classifier(args.Text)

# Print the result
print(res)
