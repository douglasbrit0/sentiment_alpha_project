# data_loader.py
import pandas as pd

def load_transcripts(path):
    df = pd.read_csv(path)  # columns: ['ticker', 'date', 'text']
    return df.dropna(subset=["text"])
