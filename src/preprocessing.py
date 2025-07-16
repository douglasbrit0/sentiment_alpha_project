# preprocessing.py
from transformers import AutoTokenizer
import re

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)  # remove speaker tags
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_texts(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
