# model_trainer.py
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def fine_tune(model_name, df_labeled):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    df_labeled['text'] = df_labeled['text'].apply(clean_text)
    dataset = Dataset.from_pandas(df_labeled[['text', 'label']])
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # use a split in production
    )

    trainer.train()
    return model
