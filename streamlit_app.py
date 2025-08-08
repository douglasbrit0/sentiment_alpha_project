# streamlit_app.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from src.interpretability import generate_shap_explainer, plot_shap_text

st.set_page_config(page_title="Sentiment SHAP Explorer")

st.title("📈 SHAP Visualizer for Sentiment Model")

# Load model + tokenizer + explainer
with st.spinner("Loading model and explainer..."):
    model_path = "models/distilbert_finetuned"
    explainer, model, tokenizer = generate_shap_explainer(model_path)

# User input
text = st.text_area("Input a sentence:", "The CEO's outlook was highly optimistic.")

if st.button("Explain"):
    with st.spinner("Running SHAP..."):
        plot_shap_text(explainer, text, tokenizer)
