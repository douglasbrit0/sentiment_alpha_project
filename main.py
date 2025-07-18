import argparse
from src.model_trainer import fine_tune
from src.model_uploader import upload_to_huggingface
from src.model_loader import load_model
import streamlit.web.cli as stcli
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Sentiment Alpha Signal Pipeline")
    
    parser.add_argument("--train", action="store_true", help="Fine-tune the model on labeled data")
    parser.add_argument("--upload", action="store_true", help="Upload model to Hugging Face Hub")
    parser.add_argument("--app", action="store_true", help="Run the Streamlit demo app")

    parser.add_argument("--model_path", type=str, default="models/distilbert_finetuned")
    parser.add_argument("--csv_path", type=str, default="data/examples.csv")
    parser.add_argument("--repo_name", type=str, help="HF repo name (e.g., douglasbrit0/sentiment-alpha-model)")
    parser.add_argument("--hf_token", type=str, help="Hugging Face access token")

    args = parser.parse_args()

    if args.train:
        fine_tune("distilbert-base-uncased", args.csv_path, args.model_path)

    if args.upload:
        if not args.repo_name or not args.hf_token:
            print("❌ Please provide --repo_name and --hf_token for upload.")
            return
        url = upload_to_huggingface(args.model_path, args.repo_name, args.hf_token)
        print(f"✅ Model uploaded to: {url}")

    if args.app:
        sys.argv = ["streamlit", "run", "app/streamlit_app.py"]
        sys.exit(stcli.main())

if __name__ == "__main__":
    main()
