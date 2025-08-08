# src/interpretability.py
import tempfile
import uuid
import os
import shap
import torch
import numpy as np
import streamlit as st
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformersTextWrapper:
    """
    Wraps a tokenizer and model so they can be used with SHAP's Explainer.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.astype(str).tolist()
        
        encoded = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.numpy()

def generate_shap_explainer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    wrapped_model = TransformersTextWrapper(tokenizer, model)
    return shap.Explainer(wrapped_model, tokenizer), model, tokenizer


def plot_shap_text(explainer: shap.Explainer, text: str, tokenizer):
    shap_values = explainer([text])
    clean_text = tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
    st.markdown(f"**Preprocessed Input:** `{clean_text}`")

    # Generate SHAP HTML
    html_obj = shap.plots.text(shap_values[0], display=False)
    html_str = html_obj.data if hasattr(html_obj, "data") else str(html_obj)

    # Write to temporary file
    tmp_dir = tempfile.gettempdir()
    tmp_filename = f"shap_output_{uuid.uuid4().hex}.html"
    tmp_filepath = os.path.join(tmp_dir, tmp_filename)

    with open(tmp_filepath, "w", encoding="utf-8") as f:
        f.write(html_str)

    # Embed using an iframe
    with open(tmp_filepath, "r", encoding="utf-8") as f:
        html_code = f.read()
        st.components.v1.html(html_code, height=600, scrolling=True)


