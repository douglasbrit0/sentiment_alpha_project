# inference.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def infer_sentiment(model_name, model_path, texts):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    return scores
