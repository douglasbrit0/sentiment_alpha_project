# Sentiment-Based Alpha Signal Generation

Independent NLP research project using fine-tuned transformer models on earnings call transcripts to generate sentiment-based alpha signals.

## Features
- Fine-tuning FinBERT/DistilBERT
- Sentiment scoring on earnings calls
- Long-short backtesting with Zipline
- Interpretability using SHAP & attention maps

## Structure

```text
sentiment_alpha_project/
├── data/              
├── models/             
├── outputs/            
├── src/
│   ├── data_loader.py    
│   ├── preprocessing.py  
│   ├── model_trainer.py  
│   ├── inference.py      
│   └── interpretability.py 
├── config.yaml         
└── main.py             
```

## TO-DO
- Add labeled training data
- Integrate with Zipline
- Build dashboard for signal visualization
