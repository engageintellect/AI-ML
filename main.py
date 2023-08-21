from transformers import pipeline

document_qa = pipeline(model="impira/layoutlm-document-qa")
res = document_qa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="Who is this shipping to?",
)


print(res)