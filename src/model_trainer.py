from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd

def fine_tune(model_name: str, csv_path: str, output_dir: str):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Load and prepare data
    df = pd.read_csv(csv_path)  # expected: 'text', 'label' columns
    df = df[df['label'].isin(['positive', 'neutral', 'negative'])]  # clean noise

    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['label'].map(label_map)

    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy='epoch',
        logging_dir='./logs',
        learning_rate=2e-5,
        evaluation_strategy='no',
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
