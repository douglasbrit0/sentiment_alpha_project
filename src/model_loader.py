from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(path_or_hub_id: str):
    """
    Loads tokenizer and model from either local path or Hugging Face Hub ID

    Args:
        path_or_hub_id (str): Path to the model folder or Hugging Face repo ID
    
    Returns:
        tokenizer (AutoTokenizer): The tokenizer for the model
        model (AutoModelForSequenceClassification): The fine-tuned model
    """
    tokenizer = AutoTokenizer.from_pretrained(path_or_hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_hub_id)
    return tokenizer, model