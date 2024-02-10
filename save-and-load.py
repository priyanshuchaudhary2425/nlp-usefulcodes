# Use this for refrence only

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch

# Define your training arguments
args = TrainingArguments(
    output_dir="./bert-finetuned-IMDB",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True
)

# Initialize the Trainer with your model, training arguments, etc.
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model locally
model.save_pretrained("./saved-model")

# Load the saved model for further training
loaded_model = AutoModelForSequenceClassification.from_pretrained("./saved-model")

# Define new training arguments if needed
args_loaded_model = TrainingArguments(
    output_dir="./bert-finetuned-IMDB-continued",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False
)

# Initialize a new Trainer with the loaded model and new training arguments
trainer_loaded_model = Trainer(
    model=loaded_model,
    args=args_loaded_model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the loaded model further
trainer_loaded_model.train()
