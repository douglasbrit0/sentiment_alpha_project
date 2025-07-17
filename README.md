# 📊 Sentiment-Based Alpha Signal Generation

Fine-tuned transformer models (DistilBERT/FinBERT) on earnings call transcripts to extract sentiment-based features for quantitative finance applications. Includes backtesting, interpretability, and a deployable Streamlit demo.

---

## 🚀 Features

- 🔍 Fine-tune transformers on financial sentiment data (Financial PhraseBank, earnings calls)
- 📈 Run sentiment-driven long-short strategies using Zipline (coming soon)
- 🧠 Visualize model decisions with attention & SHAP
- ☁️ Publish models to Hugging Face and load for inference
- 🖥️ Streamlit demo for interactive analysis

---

## 🧱 Project Structure

```text
sentiment_alpha_project/
├── src/
│   ├── data_loader.py         # Load & clean text data
│   ├── preprocessing.py       # Tokenization logic
│   ├── model_trainer.py       # Fine-tune DistilBERT/FinBERT
│   ├── inference.py           # Predict on new earnings calls
│   ├── model_loader.py        # Load from disk or Hugging Face
│   ├── model_uploader.py      # Upload model to Hugging Face Hub
│   └── interpretability.py    # SHAP and attention visualization
├── app/
│   └── streamlit_app.py       # Demo UI for sentiment analysis
├── data/                      # CSVs of training/inference data
├── models/                    # Local fine-tuned models (gitignored)
├── main.py                    # CLI to train/infer
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 📦 Setup

```bash
git clone https://github.com/douglasbrit0/sentiment_alpha_project.git
cd sentiment_alpha_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

##🧪 Fine-Tune Model

```bash
python main.py
```

Labeled CSV must have columns: text,label
Labels: positive, neutral, negative

---

## 🖥️ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## ☁️ Push Model to Hugging Face

```python
from src.model_uploader import upload_to_hub
upload_to_hub("models/distilbert_finetuned", "your-username/sentiment-alpha-model", "your_hf_token")
```

---

## 🧠 TODO
 *Backtesting module

 *Upload attention/SHAP explainers

 *Extend dataset beyond Financial PhraseBank

---


## 🏷️ Tags
transformers · finance · nlp · sentiment-analysis · alpha-signal · earnings-calls

---
