from transformers import pipeline

summarizer = pipeline("summarization")
summary = summarizer("The artificial intelligence (AI) revolution is upon us, and companies must prepare to adapt to this change. It is important to make an inventory of the current skills within the company to identify which additional skills the employees need to learn. The company does well in developing an AI strategy to outline the areas where AI is most effective, whether in a product or a service. Failing to act inevitably means falling behind. The training should include an introduction to AI, its capabilities, and its shortcomings (AI is only as good as its training data). This article gives a view of the current state of AI and what lies ahead.", min_length=5, max_length=20)
print(summary)
