from src.model_trainer import fine_tune

fine_tune(
    model_name="distilbert-base-uncased",
    csv_path="data/examples.csv",
    output_dir="models/distilbert_finetuned"
)
